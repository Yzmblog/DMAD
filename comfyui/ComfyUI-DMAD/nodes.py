"""DMAD nodes for ComfyUI: the re-noise multistep sampler the DMAD students of MiniMax-H3 were trained with, and
their sigma grid.

The students are few-step generators trained with the DMD-style multistep rule: at every step the model predicts the
clean sample x0 from x_t, and the next input is x0 re-noised with *fresh* noise at the next sigma,
x_{t'} = (1 - sigma') x0 + sigma' eps. ComfyUI's stock samplers (Euler, ...) instead follow the ODE from x_t, which
is not what the students were trained for; this sampler reproduces the rule exactly.

ComfyUI carries the H3 audio stream on the video sigma schedule (scaled by shift / audio_shift, see
`ModelSamplingAV`), so the rule applied to the packed latent is the right rule for both streams.
"""

import torch
from tqdm.auto import trange

import comfy.nested_tensor
import comfy.samplers
from comfy_api.latest import ComfyExtension, io


def _fresh_noise(like, generator):
    if getattr(like, "is_nested", False):
        return comfy.nested_tensor.NestedTensor(
            [torch.randn(t.shape, dtype=t.dtype, device=t.device, generator=generator) for t in like.unbind()]
        )
    return torch.randn(like.shape, dtype=like.dtype, device=like.device, generator=generator)


@torch.no_grad()
def sample_dmad(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Re-noise multistep sampling: x0 = model(x_i, sigma_i); x_{i+1} = (1 - sigma_{i+1}) x0 + sigma_{i+1} * fresh noise.

    `model` is ComfyUI's denoiser wrapper (returns x0), `sigmas` the flow sigmas (1 -> 0). The fresh noise is drawn
    from a generator seeded with the sampling seed, so a run is reproducible.
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed")
    generator = None if seed is None else torch.Generator(device=x.device).manual_seed(seed)
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        s_next = sigmas[i + 1]
        if s_next <= 0:
            x = denoised
        else:
            x = denoised * (1.0 - s_next) + _fresh_noise(denoised, generator) * s_next
    return x


def dmad_sigmas(steps, shift):
    """Shifted linear grid from 1 to 0 with `steps` model evaluations: sigma(u) = shift u / (1 + (shift - 1) u)."""
    u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    return shift * u / (1.0 + (shift - 1.0) * u)


class DMADSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DMADSampler",
            display_name="DMAD Sampler (re-noise)",
            category="sampling/custom_sampling/samplers",
            description="The re-noise multistep rule the DMAD students were trained with: denoise to x0, then re-noise "
                        "with fresh noise at the next sigma. Use with SamplerCustom / SamplerCustomAdvanced, the DMAD "
                        "LoRA at strength 1.0, cfg 1.0 and the DMAD Sigmas node.",
            inputs=[],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(comfy.samplers.KSAMPLER(sample_dmad))


class DMADSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DMADSigmas",
            display_name="DMAD Sigmas",
            category="sampling/custom_sampling/schedulers",
            description="The students' sigma grid: shifted linear from 1 to 0 (shift 12 for MiniMax-H3 video; set the "
                        "same video shift and audio shift 2.0 in ModelSamplingMiniMaxH3). 4 steps is the trained "
                        "setting; other step counts work too.",
            inputs=[
                io.Int.Input("steps", default=4, min=1, max=64),
                io.Float.Input("shift", default=12.0, min=0.01, max=100.0, step=0.01),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, steps, shift) -> io.NodeOutput:
        return io.NodeOutput(dmad_sigmas(steps, shift))


class DMADExtension(ComfyExtension):
    async def get_node_list(self):
        return [DMADSampler, DMADSigmas]


async def comfy_entrypoint() -> DMADExtension:
    return DMADExtension()
