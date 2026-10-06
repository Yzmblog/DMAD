"""DMAD training on ImageNet-64 (EDM teacher, one-step generator).

Each iteration: one generator update (GAN losses from the two-head critic, see edm_guidance.py), then one critic
update on the same generated batch. The generator starts from the EDM teacher; real images (T) come from the
ImageNet LMDB and teacher samples (Q) from an LMDB of teacher generations (main/edm/generate_teacher_samples.py).

Checkpoints are saved every --log_iters steps. Besides the accelerate state, each checkpoint holds the EMA
generator(s): pytorch_model_ema.bin (--generator_ema_kimg) and pytorch_model_ema_{K}.bin for every K in
--generator_ema_kimg_list. FID is computed by a separate process (main/edm/test_folder_edm.py) that watches
--cache_dir.
"""
import argparse
import os
import shutil
import time

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler

from main.data.lmdb_dataset import LMDBDataset
from main.edm.edm_unified_model import EDMUniModel
from main.utils import EMA, cycle, prepare_images_for_saving


class Trainer:
    def __init__(self, args):
        self.args = args
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="no",
            log_with="wandb",
            project_config=ProjectConfiguration(logging_dir=args.output_path),
        )
        set_seed(args.seed + accelerator.process_index)
        print(accelerator.state)

        if accelerator.is_main_process:
            self.output_path = os.path.join(args.output_path, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(self.output_path, exist_ok=False)
            if args.cache_dir != "":
                self.cache_dir = os.path.join(args.cache_dir, f"time_{int(time.time())}_seed{args.seed}")
                os.makedirs(self.cache_dir, exist_ok=False)

        self.model = EDMUniModel(args, accelerator)
        self.step = 0

        # real images (T) and teacher samples (Q)
        real_dataloader = torch.utils.data.DataLoader(
            LMDBDataset(args.real_image_path), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers
        )
        self.real_image_dataloader = cycle(accelerator.prepare(real_dataloader))
        teacher_dataloader = torch.utils.data.DataLoader(
            LMDBDataset(args.teacher_image_path), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers
        )
        self.teacher_image_dataloader = cycle(accelerator.prepare(teacher_dataloader))

        self.optimizer_guidance = torch.optim.AdamW(
            [p for p in self.model.guidance_model.parameters() if p.requires_grad],
            lr=args.guidance_lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=0.01,
        )
        self.optimizer_generator = torch.optim.AdamW(
            [p for p in self.model.feedforward_model.parameters() if p.requires_grad],
            lr=args.generator_lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=0.01,
        )
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup", optimizer=self.optimizer_guidance, num_warmup_steps=args.warmup_step, num_training_steps=args.train_iters
        )
        self.scheduler_generator = get_scheduler(
            "constant_with_warmup", optimizer=self.optimizer_generator, num_warmup_steps=args.warmup_step, num_training_steps=args.train_iters
        )
        (
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_guidance,
            self.optimizer_generator, self.scheduler_guidance, self.scheduler_generator,
        ) = accelerator.prepare(
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_guidance,
            self.optimizer_generator, self.scheduler_guidance, self.scheduler_generator,
        )

        # EMAs of the generator (evaluation only). The half-life is --generator_ema_kimg thousand images, with a
        # rampup (the half-life is at most generator_ema_rampup * images seen so far).
        generator = accelerator.unwrap_model(self.model.feedforward_model)
        self.gen_emas = {args.generator_ema_kimg: EMA(generator, decay=0.999)}
        for tok in args.generator_ema_kimg_list.split(","):
            if tok.strip() and float(tok) != args.generator_ema_kimg:
                self.gen_emas[float(tok)] = EMA(generator, decay=0.999)
        for ema in self.gen_emas.values():
            ema.ema_model.to(accelerator.device)

        self.accelerator = accelerator
        self.eye_matrix = torch.eye(args.label_dim, device=accelerator.device)
        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

        if accelerator.is_main_process:
            run = wandb.init(config=args, dir=self.output_path, mode=args.wandb_mode, entity=args.wandb_entity,
                             project=args.wandb_project, name=args.wandb_name)
            print(f"run dir: {run.dir}")

    @staticmethod
    def _ema_file(kimg, primary):
        return "pytorch_model_ema.bin" if primary else f"pytorch_model_ema_{int(kimg)}.bin"

    def load(self, checkpoint_path):
        """Full resume: model, optimizer, scheduler and RNG states, and the generator EMAs."""
        self.step = int(os.path.basename(checkpoint_path.rstrip("/")).split("_")[-1])
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        primary = self.args.generator_ema_kimg
        primary_state = torch.load(os.path.join(checkpoint_path, self._ema_file(primary, True)), map_location="cpu")
        for kimg, ema in self.gen_emas.items():
            path = os.path.join(checkpoint_path, self._ema_file(kimg, kimg == primary))
            ema.ema_model.load_state_dict(torch.load(path, map_location="cpu") if os.path.isfile(path) else primary_state)

    def save(self):
        output_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        print(f"start saving checkpoint to {output_path}")
        self.accelerator.save_state(output_path)
        for kimg, ema in self.gen_emas.items():
            torch.save(ema.ema_model.state_dict(), os.path.join(output_path, self._ema_file(kimg, kimg == self.args.generator_ema_kimg)))

        if self.args.delete_ckpts:  # keep only the latest checkpoint in output_path
            for folder in os.listdir(self.output_path):
                if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{self.step:06d}":
                    shutil.rmtree(os.path.join(self.output_path, folder))
        if self.args.cache_dir != "":  # the cache keeps the last --max_checkpoint checkpoints for the FID process
            cached = os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")
            if os.path.exists(cached):
                shutil.rmtree(cached)
            shutil.copytree(output_path, cached)
            checkpoints = sorted(f for f in os.listdir(self.cache_dir) if f.startswith("checkpoint_model"))
            for folder in checkpoints[:-self.args.max_checkpoint]:
                shutil.rmtree(os.path.join(self.cache_dir, folder))
        print("done saving")

    def _draw_batch(self):
        real_dict = next(self.real_image_dataloader)
        teacher_dict = next(self.teacher_image_dataloader)
        real_train_dict = {
            "real_image": real_dict["images"] * 2.0 - 1.0,  # [0, 1] -> [-1, 1]
            "real_label": self.eye_matrix[real_dict["class_labels"].squeeze(dim=1)],
            "second_real_image": teacher_dict["images"] * 2.0 - 1.0,
            "second_real_label": self.eye_matrix[teacher_dict["class_labels"].squeeze(dim=1)],
        }
        device = self.accelerator.device
        scaled_noise = torch.randn(self.args.batch_size, 3, self.args.resolution, self.args.resolution, device=device) * self.args.conditioning_sigma
        timestep_sigma = torch.ones(self.args.batch_size, device=device) * self.args.conditioning_sigma
        labels = torch.randint(low=0, high=self.args.label_dim, size=(self.args.batch_size,), device=device, dtype=torch.long)
        return real_train_dict, scaled_noise, timestep_sigma, self.eye_matrix[labels]

    def train_one_step(self):
        self.model.train()
        accelerator = self.accelerator
        real_train_dict, scaled_noise, timestep_sigma, labels = self._draw_batch()

        # ---- generator update ----
        generator_loss_dict, generator_log_dict = self.model(
            scaled_noise, timestep_sigma, labels, real_train_dict=real_train_dict,
            compute_generator_gradient=True, generator_turn=True, guidance_turn=False,
        )
        generator_loss = (
            generator_loss_dict["gen_teacher_loss"] * self.args.teacher_loss_weight
            + generator_loss_dict["gen_real_loss"] * self.args.real_loss_weight
        )
        accelerator.backward(generator_loss)
        generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(), self.args.max_grad_norm)
        self.optimizer_generator.step()
        self.optimizer_generator.zero_grad()  # the critic also received gradients from the generator loss
        self.optimizer_guidance.zero_grad()
        self.scheduler_generator.step()

        generator = accelerator.unwrap_model(self.model.feedforward_model)
        total_batch = self.args.batch_size * accelerator.num_processes
        cur_nimg = (self.step + 1) * total_batch
        for kimg, ema in self.gen_emas.items():
            ema_nimg = kimg * 1000
            if self.args.generator_ema_rampup > 0:
                ema_nimg = min(ema_nimg, cur_nimg * self.args.generator_ema_rampup)
            ema.decay = 0.5 ** (total_batch / max(ema_nimg, 1e-8))
            ema.update(generator)

        # ---- critic update on the same generated batch ----
        guidance_loss_dict, guidance_log_dict = self.model(
            scaled_noise, timestep_sigma, labels, real_train_dict=real_train_dict,
            compute_generator_gradient=False, generator_turn=False, guidance_turn=True,
            guidance_data_dict=generator_log_dict["guidance_data_dict"],
        )
        accelerator.backward(guidance_loss_dict["critic_loss"])
        guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.args.max_grad_norm)
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.scheduler_guidance.step()
        self.optimizer_generator.zero_grad()

        if self.step % self.args.wandb_iters == 0:
            generated_image = accelerator.gather(generator_log_dict["generated_image"])
            if accelerator.is_main_process:
                log = {
                    "generated_image": wandb.Image(prepare_images_for_saving(generated_image, resolution=self.args.resolution)),
                    "generator_grad_norm": generator_grad_norm.item(),
                    "guidance_grad_norm": guidance_grad_norm.item(),
                    "critic_loss": guidance_loss_dict["critic_loss"].item(),
                    "gen_teacher_loss": generator_loss_dict["gen_teacher_loss"].item(),
                    "gen_real_loss": generator_loss_dict["gen_real_loss"].item(),
                }
                log.update({k: v.item() for k, v in guidance_log_dict.items()})
                wandb.log(log, step=self.step)
        accelerator.wait_for_everyone()

    def train(self):
        for _ in range(self.step, self.args.train_iters):
            self.train_one_step()
            if self.accelerator.is_main_process and self.step % self.args.log_iters == 0:
                self.save()
            self.accelerator.wait_for_everyone()
            self.step += 1


def parse_args():
    parser = argparse.ArgumentParser()
    # data / model
    parser.add_argument("--model_id", type=str, required=True, help="EDM teacher pickle (edm-imagenet-64x64-cond-adm.pkl)")
    parser.add_argument("--real_image_path", type=str, required=True, help="ImageNet-64 LMDB (real images, T)")
    parser.add_argument("--teacher_image_path", type=str, required=True, help="LMDB of teacher samples (Q)")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--label_dim", type=int, default=1000)
    parser.add_argument("--use_fp16", action="store_true", help="bf16 UNet forward")
    parser.add_argument("--num_workers", type=int, default=3)
    # optimization
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--train_iters", type=int, default=1000001)
    parser.add_argument("--batch_size", type=int, default=48, help="per GPU")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--generator_lr", type=float, default=2e-6)
    parser.add_argument("--guidance_lr", type=float, default=2e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--warmup_step", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=int, default=10)
    parser.add_argument("--teacher_loss_weight", type=float, default=3e-3, help="weight of the teacher-head generator loss")
    parser.add_argument("--real_loss_weight", type=float, default=3e-3, help="weight of the real-head generator loss")
    # critic
    parser.add_argument("--critic_spectral_norm", action="store_true", help="frozen-gain spectral norm on the critic convs")
    parser.add_argument("--generator_spectral_norm", action="store_true", help="frozen-gain spectral norm on the generator encoder convs")
    parser.add_argument("--critic_freeze_backbone", action="store_true", help="freeze the critic's UNet encoder (heads train)")
    parser.add_argument("--pretrained_critic", type=str, default="", help="e.g. vgg16_bn,tf_efficientnet_lite0: frozen pretrained-feature critic")
    parser.add_argument("--critic_clean", action="store_true", help="feed the critic clean (un-noised) images")
    parser.add_argument("--critic_max_timestep", type=int, default=None, help="cap of the critic's noise-level index (0-999)")
    parser.add_argument("--teacher_gap_route", action="store_true", help="weight the teacher loss per noise band by the online gap")
    parser.add_argument("--gap_tau", type=float, default=0.5)
    parser.add_argument("--r1_gamma", type=float, default=0.0)
    parser.add_argument("--r1_batch_size", type=int, default=16)
    # noise schedule (EDM)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    parser.add_argument("--min_step_percent", type=float, default=0.02)
    parser.add_argument("--max_step_percent", type=float, default=0.98)
    # EMA
    parser.add_argument("--generator_ema_kimg", type=float, default=1000, help="EMA half-life in thousand images")
    parser.add_argument("--generator_ema_rampup", type=float, default=0.05)
    parser.add_argument("--generator_ema_kimg_list", type=str, default="", help="extra EMAs, e.g. 8000,16000")
    # checkpoints / logging
    parser.add_argument("--log_iters", type=int, default=500, help="checkpoint interval")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="resume from this checkpoint folder")
    parser.add_argument("--delete_ckpts", action="store_true", help="keep only the latest checkpoint in output_path")
    parser.add_argument("--cache_dir", type=str, default="", help="checkpoint copies for the FID process")
    parser.add_argument("--max_checkpoint", type=int, default=20, help="checkpoints kept in cache_dir")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_iters", type=int, default=100)
    args = parser.parse_args()
    if args.critic_clean:
        assert not args.teacher_gap_route, "gap routing needs noised critic inputs"
    return args


if __name__ == "__main__":
    Trainer(parse_args()).train()
