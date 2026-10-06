import os

import torch
from diffusers.utils import convert_state_dict_to_diffusers
from loguru import logger
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

from lightx2v_train.runtime.distributed import is_main_process
from lightx2v_train.runtime.fsdp import is_fsdp2_module
from lightx2v_train.utils.utils import get_running_dtype


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.running_dtype = get_running_dtype(config["model"]["running_dtype"])
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.vae = None

    def load_components(self, transformer_only=False, reference_model=None):
        raise NotImplementedError

    def denoiser_module(self):
        raise NotImplementedError(f"{self.__class__.__name__} must define denoiser_module().")

    def add_lora(self, rank, alpha, target_modules):
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.denoiser_module().add_adapter(lora_config)

    def set_lora_trainable(self):
        denoiser = self.denoiser_module()
        denoiser.requires_grad_(False)
        denoiser.train()
        for name, param in denoiser.named_parameters():
            param.requires_grad = "lora" in name

    def set_full_trainable(self):
        denoiser = self.denoiser_module()
        denoiser.requires_grad_(True)
        denoiser.train()

    def trainable_parameters(self):
        return (p for p in self.denoiser_module().parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        denoiser = self.denoiser_module()
        if hasattr(denoiser, "enable_gradient_checkpointing"):
            denoiser.enable_gradient_checkpointing()

    def is_fsdp2_wrapped(self):
        return is_fsdp2_module(self.denoiser_module())

    def fsdp2_state_module(self):
        return self.denoiser_module()

    def set_fsdp2_gradient_sync(self, enabled):
        denoiser = self.denoiser_module()
        if hasattr(denoiser, "set_requires_gradient_sync"):
            denoiser.set_requires_gradient_sync(enabled)
        if hasattr(denoiser, "set_is_last_backward"):
            denoiser.set_is_last_backward(enabled)

    def fsdp2_shard_plan(self, fsdp_config):
        raise NotImplementedError(f"{self.__class__.__name__} must define fsdp2_shard_plan().")

    def log_model_structure(self):
        logger.info("[model] class={}", self.__class__.__name__)
        text_encoder = getattr(getattr(self, "text_pipeline", None), "text_encoder", None)
        if text_encoder is not None:
            logger.info("[model] text_encoder structure:\n{}", text_encoder)
        if self.vae is not None:
            logger.info("[model] vae structure:\n{}", self.vae)
        logger.info("[model] denoiser structure:\n{}", self.denoiser_module())

    def denoise(self, denoiser_input, timesteps, condition):
        raise NotImplementedError

    def save_lora_weights(self, save_dir, adapter_name=None, weights_subdir=None):
        peft_state_dict = self._get_lora_state_dict_for_save(adapter_name=adapter_name)
        if not is_main_process():
            return

        output_dir = os.path.join(save_dir, weights_subdir) if weights_subdir else save_dir
        os.makedirs(output_dir, exist_ok=True)
        lora_state_dict = convert_state_dict_to_diffusers(peft_state_dict)
        if hasattr(self.pipeline_cls, "save_lora_weights"):
            self.pipeline_cls.save_lora_weights(output_dir, lora_state_dict, safe_serialization=True)
        else:
            save_file(lora_state_dict, os.path.join(output_dir, "pytorch_lora_weights.safetensors"))

    def _get_lora_state_dict_for_save(self, adapter_name=None):
        denoiser = self.denoiser_module()
        peft_kwargs = {} if adapter_name is None else {"adapter_name": adapter_name}
        if not is_fsdp2_module(denoiser):
            return get_peft_model_state_dict(denoiser, **peft_kwargs)

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=False,
            strict=False,
        )
        state_dict, _ = get_state_dict(denoiser, (), options=options)
        if not is_main_process():
            return {}
        return get_peft_model_state_dict(denoiser, state_dict=state_dict, **peft_kwargs)


