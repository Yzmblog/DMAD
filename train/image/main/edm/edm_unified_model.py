# A single unified model that wraps both the generator and the critic.
import copy
import pickle

import dnnlib
import torch
from torch import nn

from main.edm.edm_guidance import EDMGuidance, apply_scaled_spectral_norm
from main.edm.edm_network import get_edm_network


def load_teacher_unet(args):
    """The pretrained EDM teacher; it initializes both the generator and the critic backbone."""
    with dnnlib.util.open_url(args.model_id) as f:
        temp_edm = pickle.load(f)["ema"]
    unet = get_edm_network(args)
    unet.load_state_dict(temp_edm.state_dict(), strict=True)
    del unet.model.map_augment
    unet.model.map_augment = None
    return unet


class EDMUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        teacher = load_teacher_unet(args)
        self.guidance_model = EDMGuidance(args, teacher)
        self.feedforward_model = copy.deepcopy(teacher)
        self.feedforward_model.requires_grad_(True)
        del teacher

        if args.critic_spectral_norm:
            self.guidance_model.apply_critic_spectral_norm()
        if args.generator_spectral_norm:  # frozen-gain spectral norm on the generator encoder convs
            for module in list(self.feedforward_model.model.enc.modules()):
                apply_scaled_spectral_norm(module)
        self.guidance_model.remove_critic_decoder()
        self.accelerator = accelerator

    def forward(self, scaled_noisy_image, timestep_sigma, labels, real_train_dict=None,
                compute_generator_gradient=False, generator_turn=False, guidance_turn=False, guidance_data_dict=None):
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn)
        if generator_turn:
            if not compute_generator_gradient:
                with torch.no_grad():
                    generated_image = self.feedforward_model(scaled_noisy_image, timestep_sigma, labels)
            else:
                generated_image = self.feedforward_model(scaled_noisy_image, timestep_sigma, labels)

            if compute_generator_gradient:
                # the critic receives no gradient from the generator loss
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True, guidance_turn=False,
                    generator_data_dict={"image": generated_image, "label": labels, "real_train_dict": real_train_dict},
                )
                self.guidance_model.requires_grad_(True)
            else:
                loss_dict, log_dict = {}, {}
            log_dict["generated_image"] = generated_image.detach()
            log_dict["guidance_data_dict"] = {"image": generated_image.detach(), "label": labels.detach(), "real_train_dict": real_train_dict}
        else:
            assert guidance_data_dict is not None
            loss_dict, log_dict = self.guidance_model(generator_turn=False, guidance_turn=True, guidance_data_dict=guidance_data_dict)
        return loss_dict, log_dict
