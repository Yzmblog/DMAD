"""Log the scalar diagnostics the DMAD model puts in output_batch (keys starting with "dmad"): critic accuracies,
generator-side logits and the gap-routing state. Values are averaged over the logging window (each key only appears
in its own generator/critic phase) and logged from rank 0."""

import torch
import wandb

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel


class DMADScalars(EveryN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acc = {}

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        for k, v in output_batch.items():
            if k.startswith("dmad") and torch.is_tensor(v) and v.numel() == 1:
                self._acc.setdefault(k, []).append(float(v))
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self._acc and wandb.run:
            wandb.log({f"dmad/{k}": sum(vs) / len(vs) for k, vs in self._acc.items()}, step=iteration)
        self._acc = {}
