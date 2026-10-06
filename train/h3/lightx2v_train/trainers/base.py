import torch
from diffusers.optimization import get_scheduler
from loguru import logger

from lightx2v_train.runtime.checkpoint import find_latest_checkpoint, parse_checkpoint_iteration
from lightx2v_train.runtime.distributed import get_world_size
from lightx2v_train.runtime.monitor import build_monitor
from lightx2v_train.runtime.parallel import apply_parallel
from lightx2v_train.utils.utils import get_running_dtype


class BaseTrainer:
    """Shared trainer state: config, optimizer/lr-scheduler builders, model parallel setup, logging and resume lookup."""

    def __init__(self, config):
        self.config = config
        self.model_config = self.config["model"]
        self.training_config = self.config["training"]
        self.running_dtype = get_running_dtype(self.model_config["running_dtype"])
        self.train_type = self._resolve_train_type()
        self.gradient_checkpointing = self.training_config.get("gradient_checkpointing", True)

        optimizer_config = self._get_optimizer_config()
        self.optimizer_learning_rate = float(optimizer_config.get("learning_rate", 1e-4))
        self.optimizer_adam_beta1 = float(optimizer_config.get("adam_beta1", 0.9))
        self.optimizer_adam_beta2 = float(optimizer_config.get("adam_beta2", 0.999))
        self.optimizer_weight_decay = float(optimizer_config.get("weight_decay", 0.01))
        self.optimizer_adam_epsilon = float(optimizer_config.get("adam_epsilon", 1e-8))

        self.lr_scheduler_name = self.training_config.get("lr_scheduler", "constant")
        self.lr_warmup_iters = self.training_config["lr_warmup_iters"]
        self.max_train_iters = self.training_config["max_train_iters"]
        self.output_train_dir = self.training_config["output_dir"]
        self.gradient_accumulation_iters = self.training_config["gradient_accumulation_iters"]
        self.max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
        self.save_every_iters = self.training_config["save_every_iters"]
        self.save_total_limit = self.training_config["save_total_limit"]

        logging_config = self.config.get("logging", {})
        self.train_log_every_iters = max(1, int(logging_config.get("train_log_every_iters", 10)))
        self.monitor = build_monitor(self.config)
        self.auto_resume = self.config.get("resume", {}).get("auto_resume", False)

    def _resolve_train_type(self):
        raise NotImplementedError

    def _get_optimizer_config(self):
        raise NotImplementedError

    def _setup_trainable_model(self, model):
        raise NotImplementedError

    def set_model(self, model):
        self.model = model

    def log_metrics(self, metrics, step=None):
        self.monitor.log_metrics(metrics, step=step)

    def _build_optimizer(self, params, optimizer_config=None):
        if optimizer_config is None:
            optimizer_config = {
                "learning_rate": self.optimizer_learning_rate,
                "adam_beta1": self.optimizer_adam_beta1,
                "adam_beta2": self.optimizer_adam_beta2,
                "weight_decay": self.optimizer_weight_decay,
                "adam_epsilon": self.optimizer_adam_epsilon,
            }
        return torch.optim.AdamW(
            params,
            lr=float(optimizer_config.get("learning_rate", 1e-4)),
            betas=(float(optimizer_config.get("adam_beta1", 0.9)), float(optimizer_config.get("adam_beta2", 0.999))),
            weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
            eps=float(optimizer_config.get("adam_epsilon", 1e-8)),
        )

    def _build_lr_scheduler(self, optimizer, num_training_steps=None, num_warmup_steps=None):
        return get_scheduler(
            self.lr_scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_iters if num_warmup_steps is None else num_warmup_steps,
            num_training_steps=self.max_train_iters if num_training_steps is None else num_training_steps,
        )

    def setup(self):
        self._setup_trainable_model(self.model)
        apply_parallel(self.model, self.config)
        if self.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        self.model.log_model_structure()
        self.trainable_params = list(self.model.trainable_parameters())
        self.optimizer = self._build_optimizer(self.trainable_params)
        self.lr_scheduler = self._build_lr_scheduler(self.optimizer)

    def _validate_checkpoint_metadata(self, state, state_path, resume_ckpt_path):
        checkpoint_world_size = state.get("world_size")
        if checkpoint_world_size != get_world_size():
            raise RuntimeError(f"Cannot resume checkpoint saved with world_size={checkpoint_world_size} using world_size={get_world_size()}: {state_path}")
        expected_iteration = parse_checkpoint_iteration(resume_ckpt_path)
        if state.get("iteration") != expected_iteration:
            raise RuntimeError(f"Checkpoint iteration {state.get('iteration')} in {state_path} does not match {resume_ckpt_path}")

    def _resolve_resume(self):
        if not self.auto_resume:
            return None, 0
        ckpt_path, current_iter = find_latest_checkpoint(self.output_train_dir)
        if ckpt_path is None:
            logger.info("No checkpoint in '{}'; starting from scratch.", self.output_train_dir)
        else:
            logger.info("Resuming from {} (iteration {})", ckpt_path, current_iter)
        return ckpt_path, current_iter

    def train(self):
        raise NotImplementedError
