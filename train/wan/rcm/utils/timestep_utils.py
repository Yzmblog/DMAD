import math
from typing import Optional, Sequence, Tuple, Union

import torch


def shift_rf_time(u: torch.Tensor, shift: float) -> torch.Tensor:
    if shift <= 0:
        return u
    return shift * u / (1 + (shift - 1) * u)


def inverse_shift_rf_time(rf_t: torch.Tensor, shift: float) -> torch.Tensor:
    if shift <= 0:
        return rf_t
    return rf_t / (shift - (shift - 1.0) * rf_t)


def sigma_to_rf_time(sigma: torch.Tensor) -> torch.Tensor:
    return sigma / (sigma + 1)


def rf_to_sigma(rf_t: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(rf_t.dtype).eps
    rf_t = rf_t.clamp(min=0.0, max=1.0 - eps)
    return rf_t / (1 - rf_t)


def sigma_to_trig_time(sigma: torch.Tensor) -> torch.Tensor:
    return torch.arctan(sigma)


def trig_to_sigma(trig_t: torch.Tensor) -> torch.Tensor:
    return torch.tan(trig_t)


def rf_to_trig_time(rf_t: torch.Tensor) -> torch.Tensor:
    return sigma_to_trig_time(rf_to_sigma(rf_t))


def trig_to_rf_time(trig_t: torch.Tensor) -> torch.Tensor:
    return sigma_to_rf_time(trig_to_sigma(trig_t))


def _normalize_sample_shape(shape: Union[int, Sequence[int], torch.Size]) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    if isinstance(shape, torch.Size):
        return tuple(shape)
    if isinstance(shape, Sequence):
        return tuple(int(dim) for dim in shape)
    raise TypeError(f"Unsupported sample shape type: {type(shape)}")


def _clamp_open_unit_interval(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    return x.clamp(min=eps, max=1.0 - eps)


def _rf_logit(rf_t: torch.Tensor) -> torch.Tensor:
    rf_t = _clamp_open_unit_interval(rf_t)
    return torch.log(rf_t) - torch.log1p(-rf_t)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _normal_icdf(p: torch.Tensor) -> torch.Tensor:
    p = _clamp_open_unit_interval(p)
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


class LogNormal:
    """Log-normal sampler returning RF-domain time."""

    output_domain = "rf"

    def __init__(self, p_mean: float = 0.0, p_std: float = 1.0, **kwargs):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(
        self, shape: Union[int, Sequence[int], torch.Size], device: Union[torch.device, str] = "cuda", dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        sample_shape = _normalize_sample_shape(shape)
        log_sigma = torch.randn(sample_shape, device=device).to(dtype) * self.p_std + self.p_mean
        sigma = torch.exp(log_sigma)
        return sigma_to_rf_time(sigma)

    def cdf_rf(self, rf_t: torch.Tensor) -> torch.Tensor:
        z = (_rf_logit(rf_t) - self.p_mean) / self.p_std
        return _normal_cdf(z)

    def icdf_rf(self, u: torch.Tensor) -> torch.Tensor:
        z = self.p_mean + self.p_std * _normal_icdf(u)
        return torch.sigmoid(z)


class UniformShift:
    """Uniform sampler on [0,1) shifted in RF domain."""

    output_domain = "rf"

    def __init__(self, shift: float = 0.0, **kwargs):
        self.shift = shift

    def __call__(
        self, shape: Union[int, Sequence[int], torch.Size], device: Union[torch.device, str] = "cuda", dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        sample_shape = _normalize_sample_shape(shape)
        u = torch.rand(sample_shape, device=device).to(dtype)
        return shift_rf_time(u, self.shift)

    def cdf_rf(self, rf_t: torch.Tensor) -> torch.Tensor:
        return inverse_shift_rf_time(rf_t.clamp(0.0, 1.0), self.shift)

    def icdf_rf(self, u: torch.Tensor) -> torch.Tensor:
        return shift_rf_time(u.clamp(0.0, 1.0), self.shift)


# ---------------------------------------------------------------------------
# Generalized FoPP (Frame-oriented Probability Propagation) monotone chunk-time sampler, used in AR-Diffusion and SkyReels-V2
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_fopp_rf_chunk_times(
    sampler,
    batch_size: int,
    num_chunks: int,
    device: Union[torch.device, str] = "cuda",
    dtype: torch.dtype = torch.float64,
    anchor_chunk_idx: Optional[torch.Tensor] = None,
    anchor_rf_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Exact continuous FoPP sampling in RF domain.

    Returns a monotonically non-decreasing schedule of shape [B, num_chunks]
    where the marginal of each chunk follows the anchor sampler's distribution.
    """
    arange_B = torch.arange(batch_size, device=device)

    if anchor_chunk_idx is None:
        anchor_chunk_idx = torch.randint(0, num_chunks, (batch_size,), device=device)
    if anchor_rf_t is None:
        anchor_rf_t = sampler(shape=(batch_size,), device=device, dtype=dtype)

    anchor_rf_t = anchor_rf_t.clamp(0.0, 1.0)
    anchor_u = sampler.cdf_rf(anchor_rf_t)
    anchor_u = _clamp_open_unit_interval(anchor_u)

    u_B_chunks = torch.empty(batch_size, num_chunks, device=device, dtype=dtype)
    u_B_chunks[arange_B, anchor_chunk_idx] = anchor_u

    if num_chunks > 1:
        j = torch.arange(num_chunks, device=device)[None, :]

        left_count = anchor_chunk_idx
        exp_left = torch.empty(batch_size, num_chunks, device=device, dtype=dtype).exponential_(1.0)
        csum_left = exp_left.cumsum(dim=-1)
        left_denom = csum_left[arange_B, left_count]
        left_vals = anchor_u[:, None] * csum_left / left_denom[:, None]
        u_B_chunks = torch.where(j < left_count[:, None], left_vals, u_B_chunks)

        right_count = num_chunks - 1 - anchor_chunk_idx
        exp_right = torch.empty(batch_size, num_chunks, device=device, dtype=dtype).exponential_(1.0)
        csum_right = exp_right.cumsum(dim=-1)
        right_denom = csum_right[arange_B, right_count]
        rel_right_idx = (j - anchor_chunk_idx[:, None] - 1).clamp(min=0)
        right_cum = csum_right.gather(1, rel_right_idx)
        right_vals = anchor_u[:, None] + (1.0 - anchor_u)[:, None] * right_cum / right_denom[:, None]
        u_B_chunks = torch.where(j > anchor_chunk_idx[:, None], right_vals, u_B_chunks)

    return sampler.icdf_rf(u_B_chunks).clamp(0.0, 1.0)
