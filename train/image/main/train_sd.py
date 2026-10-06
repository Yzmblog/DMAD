"""DMAD training for SDXL (FSDP). Generator and critic are updated once per iteration each.

Checkpoints go to <output_path>/time_*/checkpoint_model_NNNNNN (latest only) and are copied to <cache_dir>/time_*/
(the last --max_checkpoint kept). Each checkpoint folder holds the generator (pytorch_model.bin), the critic
(pytorch_model_1.bin) and the generator EMA (pytorch_model_ema.bin). Evaluate with main/sdxl/eval_sdxl.py.
"""
import matplotlib
matplotlib.use("Agg")
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
from transformers import AutoTokenizer
import argparse
import shutil
import time
import os

import torch
import wandb

from main.sd_image_dataset import SDImageDatasetLMDB
from main.sd_unified_model import SDUniModel
from main.utils import SDTextDataset, cycle, prepare_images_for_saving


class Trainer:
    def __init__(self, args):
        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        accelerator_project_config = ProjectConfiguration(logging_dir=args.log_path)
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=None,
            dispatch_batches=False,
        )
        set_seed(args.seed + accelerator.process_index)
        print(accelerator.state)

        if accelerator.is_main_process:
            output_path = os.path.join(args.output_path, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(output_path, exist_ok=False)
            self.cache_dir = os.path.join(args.cache_dir, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(self.cache_dir, exist_ok=False)
            self.output_path = output_path
            os.makedirs(args.log_path, exist_ok=True)

            run = wandb.init(config=args, dir=args.log_path, mode=args.wandb_mode, entity=args.wandb_entity,
                             project=args.wandb_project, name=args.wandb_name)
            print(f"run dir: {run.dir}")
            self.wandb_folder = run.dir
            os.makedirs(self.wandb_folder, exist_ok=True)

        self.model = SDUniModel(args, accelerator)
        self.max_grad_norm = args.max_grad_norm
        self.denoising = args.denoising
        self.step = 0

        if args.ckpt_only_path is not None:
            # resume generator + critic weights (no optimizer state); the step is parsed from the folder name
            if accelerator.is_main_process:
                print(f"loading ckpt only from {args.ckpt_only_path}")
            if args.generator_spectral_norm:  # the checkpoint stores the parametrized generator
                self.model.apply_generator_spectral_norm()
            generator_path = os.path.join(args.ckpt_only_path, "pytorch_model.bin")
            guidance_path = os.path.join(args.ckpt_only_path, "pytorch_model_1.bin")
            print(self.model.feedforward_model.load_state_dict(torch.load(generator_path, map_location="cpu"), strict=True))
            print(self.model.guidance_model.load_state_dict(torch.load(guidance_path, map_location="cpu"), strict=True))
            self.step = int(args.ckpt_only_path.replace("/", "").split("_")[-1])
        else:
            if args.generator_ckpt_path is not None:
                if accelerator.is_main_process:
                    print(f"loading generator ckpt from {args.generator_ckpt_path}")
                print(self.model.feedforward_model.load_state_dict(torch.load(args.generator_ckpt_path, map_location="cpu"), strict=True))
            if args.generator_spectral_norm:
                self.model.apply_generator_spectral_norm()

        tokenizer_one = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer_2", revision=args.revision, use_fast=False
        )

        def lmdb_loader(path):
            dataset = SDImageDatasetLMDB(path, is_sdxl=True, tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two)
            dataloader = torch.utils.data.DataLoader(
                dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True
            )
            return cycle(accelerator.prepare(dataloader))

        # prompts for the generator
        dataset = SDTextDataset(args.train_prompt_path, is_sdxl=True, tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.dataloader = cycle(accelerator.prepare(dataloader))
        # real latents (T) and teacher-sample latents (Q) for the critic
        self.real_dataloader = lmdb_loader(args.real_image_path)
        self.teacher_dataloader = lmdb_loader(args.teacher_latent_path)
        # prompts for the multi-step generator (their latents are unused with --backward_simulation)
        if self.denoising:
            self.denoising_dataloader = lmdb_loader(args.real_image_path)

        self.previous_time = None

        if args.ckpt_only_path is None:
            # The randomly initialized heads differ across ranks: save rank 0's weights and reload them everywhere.
            init_ckpt_dir = os.path.join(args.output_path, f"checkpoint_model_{self.step:06d}")
            generator_path = os.path.join(init_ckpt_dir, "pytorch_model.bin")
            guidance_path = os.path.join(init_ckpt_dir, "pytorch_model_1.bin")
            if accelerator.is_main_process:
                os.makedirs(init_ckpt_dir, exist_ok=True)
                torch.save(self.model.feedforward_model.state_dict(), generator_path)
                torch.save(self.model.guidance_model.state_dict(), guidance_path)
            accelerator.wait_for_everyone()
            print(self.model.feedforward_model.load_state_dict(torch.load(generator_path, map_location="cpu"), strict=True))
            print(self.model.guidance_model.load_state_dict(torch.load(guidance_path, map_location="cpu"), strict=True))
            accelerator.wait_for_everyone()

        # only the two sub-networks are wrapped in FSDP
        self.model.feedforward_model, self.model.guidance_model = accelerator.prepare(
            self.model.feedforward_model, self.model.guidance_model
        )

        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad],
            lr=args.generator_lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=0.01,
        )
        self.optimizer_guidance = torch.optim.AdamW(
            [param for param in self.model.guidance_model.parameters() if param.requires_grad],
            lr=args.guidance_lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=0.01,
        )
        self.scheduler_generator = get_scheduler(
            "constant_with_warmup", optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step, num_training_steps=args.train_iters,
        )
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup", optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step, num_training_steps=args.train_iters,
        )
        (
            self.optimizer_generator, self.optimizer_guidance, self.scheduler_generator, self.scheduler_guidance
        ) = accelerator.prepare(
            self.optimizer_generator, self.optimizer_guidance, self.scheduler_generator, self.scheduler_guidance
        )

        # Generator EMA. parameters() are this rank's FSDP shards, so the EMA is sharded the same way and updated
        # locally; save() swaps it in to gather the full state dict.
        self.gen_ema_params = [p.detach().clone() for p in self.model.feedforward_model.parameters()]
        if args.generator_ema_ckpt_path is not None:
            # resume: load the saved EMA into the wrapped generator, copy its shards, then restore the live weights
            ema_state_dict = torch.load(args.generator_ema_ckpt_path, map_location="cpu")
            with torch.no_grad():
                live_backup = [p.detach().clone() for p in self.model.feedforward_model.parameters()]
            with FSDP.state_dict_type(
                self.model.feedforward_model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            ):
                print(self.model.feedforward_model.load_state_dict(ema_state_dict, strict=True))
            self.gen_ema_params = [p.detach().clone() for p in self.model.feedforward_model.parameters()]
            with torch.no_grad():
                for p, b in zip(self.model.feedforward_model.parameters(), live_backup):
                    p.data.copy_(b)
            del live_backup, ema_state_dict

        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.latent_resolution = args.latent_resolution
        self.grid_size = args.grid_size
        self.latent_channel = args.latent_channel
        self.max_checkpoint = args.max_checkpoint

    def fsdp_state_dict(self, model):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            return model.state_dict()

    def save(self):
        # The latest checkpoint is kept in output_path and every checkpoint is copied to cache_dir.
        # Only model weights are saved (no optimizer state).
        feedforward_state_dict = self.fsdp_state_dict(self.model.feedforward_model)
        guidance_model_state_dict = self.fsdp_state_dict(self.model.guidance_model)

        with torch.no_grad():  # swap the EMA in, gather it, restore the live weights (all ranks)
            live_backup = [p.detach().clone() for p in self.model.feedforward_model.parameters()]
            for p, s in zip(self.model.feedforward_model.parameters(), self.gen_ema_params):
                p.data.copy_(s)
        ema_state_dict = self.fsdp_state_dict(self.model.feedforward_model)
        with torch.no_grad():
            for p, b in zip(self.model.feedforward_model.parameters(), live_backup):
                p.data.copy_(b)
        del live_backup

        if self.accelerator.is_main_process:
            name = f"checkpoint_model_{self.step:06d}"
            output_path = os.path.join(self.output_path, name)
            os.makedirs(output_path, exist_ok=True)
            print(f"start saving checkpoint to {output_path}")
            torch.save(feedforward_state_dict, os.path.join(output_path, "pytorch_model.bin"))
            torch.save(guidance_model_state_dict, os.path.join(output_path, "pytorch_model_1.bin"))
            torch.save(ema_state_dict, os.path.join(output_path, "pytorch_model_ema.bin"))

            for folder in os.listdir(self.output_path):
                if folder.startswith("checkpoint_model") and folder != name:
                    shutil.rmtree(os.path.join(self.output_path, folder))

            if os.path.exists(os.path.join(self.cache_dir, name)):
                shutil.rmtree(os.path.join(self.cache_dir, name))
            shutil.copytree(output_path, os.path.join(self.cache_dir, name))
            checkpoints = sorted(folder for folder in os.listdir(self.cache_dir) if folder.startswith("checkpoint_model"))
            for folder in checkpoints[:-self.max_checkpoint]:
                shutil.rmtree(os.path.join(self.cache_dir, folder))
            print("done saving")
        del feedforward_state_dict, guidance_model_state_dict, ema_state_dict
        torch.cuda.empty_cache()

    def train_one_step(self):
        self.model.train()
        accelerator = self.accelerator

        noise = torch.randn(self.batch_size, self.latent_channel, self.latent_resolution, self.latent_resolution, device=accelerator.device)
        visual = self.step % self.wandb_iters == 0

        text_embedding = next(self.dataloader)
        denoising_dict = next(self.denoising_dataloader) if self.denoising else None
        real_train_dict = next(self.real_dataloader)

        # generator update
        generator_loss_dict, generator_log_dict = self.model(
            noise, text_embedding, visual=visual, denoising_dict=denoising_dict,
            real_train_dict=real_train_dict, generator_turn=True,
        )
        generator_loss = (
            generator_loss_dict["gen_teacher_loss"] * self.args.teacher_loss_weight
            + generator_loss_dict["gen_real_loss"] * self.args.real_loss_weight
        )
        self.accelerator.backward(generator_loss)
        generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(), self.max_grad_norm)
        self.optimizer_generator.step()

        # EMA with a half-life of generator_ema_kimg thousand images, ramped up early in training
        total_batch = self.batch_size * accelerator.num_processes
        cur_nimg = (self.step + 1) * total_batch
        ema_nimg = self.args.generator_ema_kimg * 1000
        if self.args.generator_ema_rampup > 0:
            ema_nimg = min(ema_nimg, cur_nimg * self.args.generator_ema_rampup)
        decay = 0.5 ** (total_batch / max(ema_nimg, 1e-8))
        with torch.no_grad():
            for s, p in zip(self.gen_ema_params, self.model.feedforward_model.parameters()):
                s.lerp_(p.detach(), 1.0 - decay)

        # the critic may also have received gradients; clear both
        self.optimizer_generator.zero_grad()
        self.optimizer_guidance.zero_grad()
        self.scheduler_generator.step()

        # critic update
        guidance_data_dict = {**generator_log_dict["guidance_data_dict"], "teacher_train_dict": next(self.teacher_dataloader)}
        guidance_loss_dict, guidance_log_dict = self.model(
            noise, text_embedding, visual=visual, denoising_dict=denoising_dict,
            real_train_dict=real_train_dict, guidance_turn=True, guidance_data_dict=guidance_data_dict,
        )
        guidance_loss = guidance_loss_dict["critic_loss"] * self.args.critic_loss_weight
        self.accelerator.backward(guidance_loss)
        guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.optimizer_generator.zero_grad()
        self.scheduler_guidance.step()

        generated_image = generator_log_dict["guidance_data_dict"]["image"]
        generated_image_mean = accelerator.gather(generated_image.mean()).mean()
        generated_image_std = accelerator.gather(generated_image.std()).mean()

        if visual:
            generator_log_dict["generated_image"] = accelerator.gather(generator_log_dict["generated_image"])

        if accelerator.is_main_process:
            log = {
                "generator_grad_norm": generator_grad_norm.item(),
                "guidance_grad_norm": guidance_grad_norm.item(),
                "generated_image_mean": generated_image_mean.item(),
                "generated_image_std": generated_image_std.item(),
                "gen_teacher_loss": generator_loss_dict["gen_teacher_loss"].item(),
                "gen_real_loss": generator_loss_dict["gen_real_loss"].item(),
                "critic_loss": guidance_loss_dict["critic_loss"].item(),
            }
            for k, v in guidance_log_dict.items():
                log[k] = v.item() if torch.is_tensor(v) else v
            if visual:
                with torch.no_grad():
                    grid = prepare_images_for_saving(generator_log_dict["generated_image"], resolution=self.resolution, grid_size=self.grid_size)
                log["generated_image"] = wandb.Image(grid)
            wandb.log(log, step=self.step)

        self.accelerator.wait_for_everyone()

    def train(self):
        for index in range(self.step, self.train_iters):
            self.train_one_step()
            if self.step % self.log_iters == 0:
                self.save()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                current_time = time.time()
                if self.previous_time is not None:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                self.previous_time = current_time

            self.step += 1


def parse_args():
    parser = argparse.ArgumentParser()
    # data / model
    parser.add_argument("--model_id", type=str, required=True, help="local stable-diffusion-xl-base-1.0 folder")
    parser.add_argument("--revision", type=str)
    parser.add_argument("--train_prompt_path", type=str, required=True, help="captions_laion_score6.25.pkl")
    parser.add_argument("--real_image_path", type=str, required=True, help="LMDB of real VAE latents (T)")
    parser.add_argument("--teacher_latent_path", type=str, required=True, help="LMDB of teacher-sample latents (Q)")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--latent_resolution", type=int, default=128)
    parser.add_argument("--latent_channel", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--use_fp16", action="store_true", help="bf16 autocast for the UNet forwards")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # generator
    parser.add_argument("--conditioning_timestep", type=int, default=999, help="input timestep of the one-step generator")
    parser.add_argument("--denoising", action="store_true", help="multi-step generator")
    parser.add_argument("--num_denoising_step", type=int, default=4)
    parser.add_argument("--denoising_timestep", type=int, default=1000)
    parser.add_argument("--backward_simulation", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--generator_ckpt_path", type=str, help="generator initialization (e.g. the ODE-pretrained one-step model)")
    # optimization
    parser.add_argument("--train_iters", type=int, default=26001)
    parser.add_argument("--batch_size", type=int, default=8, help="per GPU")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--generator_lr", type=float, default=5e-7)
    parser.add_argument("--guidance_lr", type=float, default=5e-7)
    parser.add_argument("--adam_beta1", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--warmup_step", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--teacher_loss_weight", type=float, default=5e-3, help="weight of the teacher-head generator loss")
    parser.add_argument("--real_loss_weight", type=float, default=5e-3, help="weight of the real-head generator loss")
    parser.add_argument("--critic_loss_weight", type=float, default=1e-2)
    # critic
    parser.add_argument("--critic_max_timestep", type=int, default=1000, help="critic noise timesteps are uniform in [0, this)")
    parser.add_argument("--critic_spectral_norm", action="store_true", help="frozen-gain spectral norm on the critic convs")
    parser.add_argument("--generator_spectral_norm", action="store_true", help="frozen-gain spectral norm on the generator encoder convs")
    parser.add_argument("--critic_freeze_backbone", action="store_true", help="freeze the critic's UNet encoder (heads train)")
    parser.add_argument("--teacher_gap_route", action="store_true", help="weight the teacher loss per noise band by the online gap")
    parser.add_argument("--gap_tau", type=float, default=0.5)
    # EMA
    parser.add_argument("--generator_ema_kimg", type=float, default=1000, help="EMA half-life in thousand images")
    parser.add_argument("--generator_ema_rampup", type=float, default=0.05)
    # checkpoints / resume / logging
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--log_iters", type=int, default=500, help="checkpoint interval")
    parser.add_argument("--max_checkpoint", type=int, default=20, help="checkpoints kept in cache_dir")
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="resume generator + critic weights from this checkpoint folder")
    parser.add_argument("--generator_ema_ckpt_path", type=str, default=None, help="with --ckpt_only_path: that folder's pytorch_model_ema.bin")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--grid_size", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
