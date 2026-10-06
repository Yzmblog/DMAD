# A single unified model that wraps both the SDXL generator and the DMAD critic
from diffusers import AutoencoderKL, UNet2DConditionModel
from accelerate.utils import broadcast
from torch import nn
import torch

from main.sd_guidance import SDGuidance, apply_scaled_spectral_norm
from main.sdxl.sdxl_text_encoder import SDXLTextEncoder
from main.utils import NoOpContext, get_x0_from_noise


class SDUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.accelerator = accelerator
        self.guidance_model = SDGuidance(args, accelerator)
        self.num_train_timesteps = self.guidance_model.num_train_timesteps
        self.num_visuals = args.grid_size * args.grid_size
        self.conditioning_timestep = args.conditioning_timestep
        self.use_fp16 = args.use_fp16
        self.gradient_checkpointing = args.gradient_checkpointing
        self.backward_simulation = args.backward_simulation

        self.denoising = args.denoising
        self.denoising_timestep = args.denoising_timestep
        self.noise_scheduler = self.guidance_model.scheduler
        self.num_denoising_step = args.num_denoising_step
        self.denoising_step_list = torch.tensor(
            list(range(self.denoising_timestep - 1, 0, -(self.denoising_timestep // self.num_denoising_step))),
            dtype=torch.long,
            device=accelerator.device,
        )
        self.timestep_interval = self.denoising_timestep // self.num_denoising_step

        self.feedforward_model = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").float()
        self.feedforward_model.requires_grad_(True)
        if self.gradient_checkpointing:
            self.feedforward_model.enable_gradient_checkpointing()

        self.text_encoder = SDXLTextEncoder(args, accelerator).to(accelerator.device)
        self.text_encoder.requires_grad_(False)
        self.add_time_ids = self.build_condition_input(args.resolution, accelerator)

        self.alphas_cumprod = self.guidance_model.alphas_cumprod.to(accelerator.device)

        # SDXL's VAE does not work in half precision
        self.vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").float().to(accelerator.device)
        self.vae.requires_grad_(False)

        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_fp16 else NoOpContext()

    def apply_generator_spectral_norm(self):
        """Frozen-gain spectral norm on the generator's encoder convs. Called by the trainer after all generator weight
        loads (the gains are frozen at the loaded weights) and before FSDP wrapping."""
        n = 0
        for module in (self.feedforward_model.down_blocks, self.feedforward_model.mid_block):
            n += apply_scaled_spectral_norm(module)
        if self.accelerator.is_local_main_process:
            print(f"generator spectral norm: {n} convs")

    def build_condition_input(self, resolution, accelerator):
        original_size = (resolution, resolution)
        target_size = (resolution, resolution)
        crop_top_left = (0, 0)

        add_time_ids = list(original_size + crop_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=accelerator.device, dtype=torch.float32)
        return add_time_ids

    def added_conditions(self, pooled_text_embedding):
        return {"time_ids": self.add_time_ids.repeat(len(pooled_text_embedding), 1), "text_embeds": pooled_text_embedding}

    def decode_image(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample.float()
        return image

    @torch.no_grad()
    def sample_backward(self, noisy_image, real_text_embedding, real_pooled_text_embedding):
        """Backward simulation: run the generator's own multi-step sampler for a random number of steps (shared across
        GPUs) and return the intermediate clean estimate and the timestep it will be re-noised to."""
        batch_size = noisy_image.shape[0]
        device = noisy_image.device
        unet_added_conditions = self.added_conditions(real_pooled_text_embedding)

        selected_step = torch.randint(low=0, high=self.num_denoising_step, size=(1,), device=device, dtype=torch.long)
        selected_step = broadcast(selected_step, from_process=0)

        # default in case the loop is skipped; overwritten by the pure-noise mask later
        generated_image = noisy_image

        for constant in self.denoising_step_list[:selected_step]:
            current_timesteps = torch.ones(batch_size, device=device, dtype=torch.long) * constant

            generated_noise = self.feedforward_model(
                noisy_image, current_timesteps, real_text_embedding, added_cond_kwargs=unet_added_conditions
            ).sample

            generated_image = get_x0_from_noise(
                noisy_image, generated_noise.double(), self.alphas_cumprod.double(), current_timesteps
            ).float()

            next_timestep = current_timesteps - self.timestep_interval
            noisy_image = self.noise_scheduler.add_noise(
                generated_image, torch.randn_like(generated_image), next_timestep
            ).to(noisy_image.dtype)

        return_timesteps = self.denoising_step_list[selected_step] * torch.ones(batch_size, device=device, dtype=torch.long)
        return generated_image, return_timesteps

    @torch.no_grad()
    def encode_real_captions(self, real_train_dict):
        real_text_embedding, real_pooled_text_embedding = self.text_encoder(real_train_dict)
        real_train_dict["text_embedding"] = real_text_embedding
        real_train_dict["unet_added_conditions"] = self.added_conditions(real_pooled_text_embedding)
        return real_train_dict

    @torch.no_grad()
    def prepare_denoising_data(self, denoising_dict, real_train_dict, noise):
        indices = torch.randint(0, self.num_denoising_step, (noise.shape[0],), device=noise.device, dtype=torch.long)
        timesteps = self.denoising_step_list.to(noise.device)[indices]

        text_embedding, pooled_text_embedding = self.text_encoder(denoising_dict)
        real_train_dict = self.encode_real_captions(real_train_dict)

        if self.backward_simulation:
            # overwrites the timesteps; uses noise uncorrelated with `noise`
            clean_images, timesteps = self.sample_backward(torch.randn_like(noise), text_embedding, pooled_text_embedding)
        else:
            clean_images = denoising_dict["images"].to(noise.device)

        noisy_image = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # the last timestep is pure noise
        pure_noise_mask = timesteps == (self.num_train_timesteps - 1)
        noisy_image[pure_noise_mask] = noise[pure_noise_mask]

        return timesteps, text_embedding, pooled_text_embedding, real_train_dict, noisy_image

    @torch.no_grad()
    def prepare_pure_generation_data(self, text_embedding, real_train_dict, noise):
        # text_embedding is a batch of tokenized prompts here
        text_embedding_output = self.text_encoder(text_embedding)
        text_embedding = text_embedding_output[0].float()
        pooled_text_embedding = text_embedding_output[1].float()
        real_train_dict = self.encode_real_captions(real_train_dict)
        return text_embedding, pooled_text_embedding, real_train_dict, noise

    def forward(self, noise, text_embedding, visual=False, denoising_dict=None, real_train_dict=None,
                generator_turn=False, guidance_turn=False, guidance_data_dict=None):
        assert generator_turn != guidance_turn

        if generator_turn:
            if self.denoising:
                # the prompts come from denoising_dict; text_embedding is ignored
                timesteps, text_embedding, pooled_text_embedding, real_train_dict, noisy_image = self.prepare_denoising_data(
                    denoising_dict, real_train_dict, noise
                )
            else:
                timesteps = torch.ones(noise.shape[0], device=noise.device, dtype=torch.long) * self.conditioning_timestep
                text_embedding, pooled_text_embedding, real_train_dict, noisy_image = self.prepare_pure_generation_data(
                    text_embedding, real_train_dict, noise
                )
            unet_added_conditions = self.added_conditions(pooled_text_embedding)

            with self.network_context_manager:
                generated_noise = self.feedforward_model(
                    noisy_image, timesteps.long(), text_embedding, added_cond_kwargs=unet_added_conditions
                ).sample

            # SDXL uses epsilon prediction
            generated_image = get_x0_from_noise(
                noisy_image.double(), generated_noise.double(), self.alphas_cumprod.double(), timesteps
            ).float()

            generator_data_dict = {
                "image": generated_image,
                "text_embedding": text_embedding,
                "unet_added_conditions": unet_added_conditions,
            }
            # The critic stays frozen through the generator backward; it is re-enabled at the start of the next critic
            # update. (Flipping requires_grad between the FSDP forward and backward breaks FSDP's post-backward hooks.)
            self.guidance_model.requires_grad_(False)
            loss_dict, log_dict = self.guidance_model(generator_turn=True, generator_data_dict=generator_data_dict)

            if visual:
                with torch.no_grad():
                    log_dict["generated_image"] = self.decode_image(generated_image[:self.num_visuals].detach())

            log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(),
                "text_embedding": text_embedding.detach(),
                "real_train_dict": real_train_dict,
                "unet_added_conditions": unet_added_conditions,
            }
            log_dict["denoising_timestep"] = timesteps

        else:
            # teacher samples (Q) arrive tokenized; encode them here, inside forward (FSDP gathers the text encoder
            # parameters only within the wrapped forward)
            with torch.no_grad():
                guidance_data_dict["teacher_train_dict"] = self.encode_real_captions(guidance_data_dict["teacher_train_dict"])
            self.guidance_model.requires_grad_(True)
            loss_dict, log_dict = self.guidance_model(guidance_turn=True, guidance_data_dict=guidance_data_dict)
        return loss_dict, log_dict
