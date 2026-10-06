"""Two-model distillation trainer: a student (generator) and a fake/critic model, alternating updates.

Subclasses implement _latent_shape(), sample_initial_latents(latent_shape) and
forward_loss(latent_shape, stage, initial_noise) for stage in {"student", "fake"}.
Checkpoints (FSDP2): student LoRA weights, critic LoRA weights (fake_lora/), trainer_state.pt and the distributed
model/optimizer state (dist_state/).
"""

import copy
import os
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime.checkpoint import prune_checkpoints
from lightx2v_train.runtime.distributed import barrier, get_world_size, is_distributed, is_main_process, reduce_mean
from lightx2v_train.runtime.parallel import apply_parallel, set_parallel_gradient_sync

from ..base import BaseTrainer

CHECKPOINT_VERSION = 2


def _lora_config(role_config):
    if role_config["train_type"] != "lora":
        return None
    config = copy.deepcopy(role_config["lora"])
    config["rank"] = int(config["rank"])
    config["alpha"] = int(config["alpha"])
    return config


class DmdTrainer(BaseTrainer):
    allowed_model_names = None
    default_lora_target_modules = None

    def _resolve_train_type(self):
        if "train_type" in self.training_config:
            raise ValueError("Use training.student.train_type and training.fake.train_type (not training.train_type).")
        return None

    def __init__(self, config):
        super().__init__(config)
        if self.allowed_model_names and self.model_config["name"] not in self.allowed_model_names:
            raise ValueError(f"{type(self).__name__} requires model.name in {sorted(self.allowed_model_names)}.")
        training = self.training_config
        self.student_config = training["student"]
        self.fake_config = training["fake"]
        self.dmd_config = training["dmd"]
        self.student_train_type = self.student_config["train_type"]
        self.fake_train_type = self.fake_config["train_type"]
        self.student_lora_config = _lora_config(self.student_config)
        self.fake_lora_config = _lora_config(self.fake_config)
        for lora_config in (self.student_lora_config, self.fake_lora_config):
            if lora_config is not None and "target_modules" not in lora_config:
                lora_config["target_modules"] = list(self.default_lora_target_modules)

        self.fake_optimizer_config = self.fake_config["optimizer"]
        self.fake_optimizer_hparams = {
            "learning_rate": self.fake_optimizer_config.get("learning_rate", self.optimizer_learning_rate),
            "adam_beta1": self.fake_optimizer_config.get("adam_beta1", self.optimizer_adam_beta1),
            "adam_beta2": self.fake_optimizer_config.get("adam_beta2", self.optimizer_adam_beta2),
            "weight_decay": self.fake_optimizer_config.get("weight_decay", self.optimizer_weight_decay),
            "adam_epsilon": self.fake_optimizer_config.get("adam_epsilon", self.optimizer_adam_epsilon),
        }
        self.num_inference_steps = int(self.dmd_config.get("num_inference_steps", 4))
        self.fake_update_ratio = max(1, int(self.dmd_config.get("fake_update_ratio", 1)))

    def _get_optimizer_config(self):
        return self.training_config["student"]["optimizer"]

    def _setup_trainable_model(self, model, role="student"):
        train_type, lora_config = (
            (self.student_train_type, self.student_lora_config) if role == "student" else (self.fake_train_type, self.fake_lora_config)
        )
        if train_type == "lora":
            model.add_lora(lora_config["rank"], lora_config["alpha"], lora_config.get("target_modules"))
            model.set_lora_trainable()
            return
        model.set_full_trainable()

    # ------------------------ setup ------------------------

    def setup(self, resume_ckpt_path=None):
        super().setup()  # student: LoRA, FSDP2, gradient checkpointing, optimizer, lr scheduler
        fake_model_config = copy.deepcopy(self.config)
        self.fake_model = build_model(fake_model_config)
        self.fake_model.load_components(transformer_only=True, reference_model=self.model)
        self._setup_trainable_model(self.fake_model, role="fake")
        apply_parallel(self.fake_model, self.config)
        if self.gradient_checkpointing:
            self.fake_model.enable_gradient_checkpointing()
        self.fake_trainable_params = list(self.fake_model.trainable_parameters())
        self.fake_optimizer = self._build_optimizer(self.fake_trainable_params, self.fake_optimizer_hparams)
        self.fake_lr_scheduler = self._build_lr_scheduler(
            self.fake_optimizer, num_warmup_steps=0, num_training_steps=max(1, self.max_train_iters * self.fake_update_ratio)
        )
        if resume_ckpt_path is not None:
            self._load_resume_state(resume_ckpt_path)
        logger.info("[train] student train_type={} fake train_type={}", self.student_train_type, self.fake_train_type)

    # ------------------------ training loop ------------------------

    def _sample_synced_int(self, low, high):
        value = torch.randint(int(low), int(high), (1,), device=self.model.device, dtype=torch.int64)
        if is_distributed():
            dist.broadcast(value, src=0)
        return int(value.item())

    def _after_student_optimizer_step(self, region):
        pass

    def train(self):
        resume_ckpt_path, current_iter = self._resolve_resume()
        self.setup(resume_ckpt_path=resume_ckpt_path)
        if is_main_process():
            os.makedirs(self.output_train_dir, exist_ok=True)
        barrier()
        grad_accum_iters = max(1, int(self.gradient_accumulation_iters))
        logger.info(
            "[train] start iter={}/{} world_size={} fake_update_ratio={}",
            current_iter, self.max_train_iters, get_world_size(), self.fake_update_ratio,
        )
        while current_iter < self.max_train_iters:
            student_result = self._train_one_stage(stage="student", grad_accum_iters=grad_accum_iters)
            loss_fake_value = 0.0
            for _ in range(self.fake_update_ratio):
                loss_fake_value += self._train_one_stage(stage="fake", grad_accum_iters=grad_accum_iters)["loss"]
            loss_fake_value /= self.fake_update_ratio

            current_iter += 1
            display_fake = reduce_mean(loss_fake_value)
            display_dmd = reduce_mean(student_result["dmd"])
            current_lr = self.lr_scheduler.get_last_lr()[0]
            if current_iter == 1 or current_iter % self.train_log_every_iters == 0 or current_iter >= self.max_train_iters:
                logger.info("[train] iter={}/{} dmd={:.6f} fake={:.6f} lr={:.8f}", current_iter, self.max_train_iters, display_dmd, display_fake, current_lr)
                self.log_metrics({"train/dmd": display_dmd, "train/fake": display_fake, "train/lr": current_lr}, step=current_iter)
            if self.save_every_iters and current_iter % self.save_every_iters == 0:
                self.save_checkpoint(current_iter, self.save_total_limit)
        logger.info("[train] finished iter={}/{}", current_iter, self.max_train_iters)

    def _train_one_stage(self, stage, grad_accum_iters):
        if stage == "student":
            optimizer, scheduler, params, set_sync = self.optimizer, self.lr_scheduler, self.trainable_params, self._set_student_gradient_sync
        else:
            optimizer, scheduler, params, set_sync = self.fake_optimizer, self.fake_lr_scheduler, self.fake_trainable_params, self._set_fake_gradient_sync

        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        metric_values = {}
        for micro_idx in range(grad_accum_iters):
            latent_shape = self._latent_shape()
            initial_noise = self.sample_initial_latents(latent_shape)
            set_sync(micro_idx == grad_accum_iters - 1)
            result = self.forward_loss(latent_shape, stage=stage, initial_noise=initial_noise)
            loss = result.pop("loss")
            for name, value in result.items():
                metric_values[name] = metric_values.get(name, 0.0) + value.item() / grad_accum_iters
            (loss / grad_accum_iters).backward()
            loss_value += loss.item() / grad_accum_iters
            del loss, result, initial_noise

        self._before_optimizer_step(params)
        torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        optimizer.step()
        if stage == "student":
            self._after_student_optimizer_step("main")
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return {"loss": loss_value, **metric_values}

    def _set_student_gradient_sync(self, enabled):
        set_parallel_gradient_sync(self.model, enabled)

    def _set_fake_gradient_sync(self, enabled):
        set_parallel_gradient_sync(self.fake_model, enabled)

    def _before_optimizer_step(self, params):
        """Hook after the backward pass(es) of a stage and before its optimizer step."""

    # ------------------------ checkpointing (FSDP2) ------------------------

    def _fake_weights_dir(self, root_dir):
        return os.path.join(root_dir, "fake_lora" if self.fake_train_type == "lora" else "fake_model")

    def _save_model_weights(self, model, save_dir, train_type):
        if train_type == "lora":
            model.save_lora_weights(save_dir)
        elif is_main_process():
            torch.save(model.denoiser_module().state_dict(), os.path.join(save_dir, "model_state.pt"))

    def _state_dicts(self):
        options = StateDictOptions(ignore_frozen_params=True, strict=False)
        student_model, student_optim = get_state_dict(self.model.fsdp2_state_module(), self.optimizer, options=options)
        fake_model, fake_optim = get_state_dict(self.fake_model.fsdp2_state_module(), self.fake_optimizer, options=options)
        state = {"student_model": student_model, "student_optimizer": student_optim, "fake_model": fake_model, "fake_optimizer": fake_optim}
        return state, options

    def save_checkpoint(self, iteration, save_total_limit):
        if not (self.model.is_fsdp2_wrapped() and self.fake_model.is_fsdp2_wrapped()):
            raise RuntimeError("checkpointing requires FSDP2 (distributed.fsdp2.enabled: true)")
        if is_main_process():
            prune_checkpoints(self.output_train_dir, save_total_limit)
        save_dir = os.path.join(self.output_train_dir, f"checkpoint-{iteration:09d}")
        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)
        barrier()
        if self.student_train_type == "lora":
            self._save_model_weights(self.model, save_dir, self.student_train_type)
        barrier()
        fake_save_dir = self._fake_weights_dir(save_dir)
        if self.fake_train_type == "lora" and is_main_process():
            os.makedirs(fake_save_dir, exist_ok=True)
        barrier()
        if self.fake_train_type == "lora":
            self._save_model_weights(self.fake_model, fake_save_dir, self.fake_train_type)
        barrier()
        config_path = self.config.get("config_path")
        if is_main_process() and config_path is not None:
            shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))

        trainer_state = {
            "iteration": iteration,
            "world_size": get_world_size(),
            "dmd_checkpoint_version": CHECKPOINT_VERSION,
            "student_train_type": self.student_train_type,
            "fake_train_type": self.fake_train_type,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "fake_lr_scheduler": self.fake_lr_scheduler.state_dict(),
        }
        dist_state_path = os.path.join(save_dir, "dist_state")
        if is_main_process():
            os.makedirs(dist_state_path, exist_ok=True)
            torch.save(trainer_state, os.path.join(save_dir, "trainer_state.pt"))
        barrier()
        state, _ = self._state_dicts()
        dcp.save(state, checkpoint_id=dist_state_path)
        barrier()
        logger.info("[train] saved checkpoint iter={} path={}", iteration, save_dir)

    def _load_resume_state(self, resume_ckpt_path):
        dist_state_path = os.path.join(resume_ckpt_path, "dist_state")
        trainer_state_path = os.path.join(resume_ckpt_path, "trainer_state.pt")
        if not os.path.exists(dist_state_path) or not os.path.exists(trainer_state_path):
            raise RuntimeError(f"resume needs dist_state/ and trainer_state.pt in {resume_ckpt_path}")
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_metadata(trainer_state, trainer_state_path, resume_ckpt_path)
        for role, current in (("student", self.student_train_type), ("fake", self.fake_train_type)):
            saved = trainer_state.get(f"{role}_train_type")
            if saved is not None and saved != current:
                raise RuntimeError(f"checkpoint {role}_train_type={saved!r} != training.{role}.train_type={current!r}")
        state, options = self._state_dicts()
        dcp.load(state, checkpoint_id=dist_state_path)
        set_state_dict(self.model.fsdp2_state_module(), self.optimizer, model_state_dict=state["student_model"],
                       optim_state_dict=state["student_optimizer"], options=options)
        set_state_dict(self.fake_model.fsdp2_state_module(), self.fake_optimizer, model_state_dict=state["fake_model"],
                       optim_state_dict=state["fake_optimizer"], options=options)
        self.lr_scheduler.load_state_dict(trainer_state["lr_scheduler"])
        self.fake_lr_scheduler.load_state_dict(trainer_state["fake_lr_scheduler"])
        logger.info("[train] resumed from {} (iteration {})", resume_ckpt_path, trainer_state["iteration"])
