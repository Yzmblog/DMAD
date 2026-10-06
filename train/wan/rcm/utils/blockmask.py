from dataclasses import dataclass, replace
from functools import partial
import os
from typing import Optional, Literal, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention, create_block_mask
from rcm.utils.attention import attention


def _compile_flex_attention():
    backend = os.environ.get("RCM_FLEX_BACKEND", "auto").lower()
    if backend not in {"auto", "triton", "flash"}:
        raise ValueError(f"Unsupported RCM_FLEX_BACKEND={backend!r}; expected one of auto, triton, flash")

    flex_fn = torch_flex_attention
    if backend != "auto":
        # PyTorch FlexAttention uses BACKEND="FLASH" for the FA4/CuTe backend.
        flex_fn = partial(torch_flex_attention, kernel_options={"BACKEND": backend.upper()})

    return torch.compile(flex_fn, dynamic=False, mode=os.environ.get("RCM_FLEX_COMPILE_MODE", "max-autotune-no-cudagraphs"))


flex_attention = _compile_flex_attention()


def flex_attention_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D, block_mask):
    return flex_attention(q_B_S_H_D.transpose(1, 2), k_B_S_H_D.transpose(1, 2), v_B_S_H_D.transpose(1, 2), block_mask=block_mask).transpose(1, 2)


def _call_magi_flex_flash(q, k, v, q_ranges, k_ranges):
    try:
        from magi_attention.api import flex_flash_attn_func
    except ImportError as exc:
        raise ImportError(
            "RCM_ATTENTION_BACKEND=magi requires the optional `magi_attention` package. "
            "Install SandAI MagiAttention or use RCM_ATTENTION_BACKEND=flex."
        ) from exc

    B, Q, H, D = q.shape
    KV = k.shape[1]
    num_ranges = q_ranges.shape[0]

    q_offsets = (torch.arange(B, device=q.device, dtype=torch.int32) * Q).view(B, 1, 1)
    k_offsets = (torch.arange(B, device=k.device, dtype=torch.int32) * KV).view(B, 1, 1)
    q_ranges = (q_ranges.view(1, num_ranges, 2) + q_offsets).reshape(B * num_ranges, 2).contiguous()
    k_ranges = (k_ranges.view(1, num_ranges, 2) + k_offsets).reshape(B * num_ranges, 2).contiguous()
    # MagiAttention uses 0 for FULL slices. The local mask builder already splits
    # causal patterns into full rectangles, so every emitted slice is FULL.
    attn_type_map = torch.zeros((B * num_ranges,), device=q.device, dtype=torch.int32)

    q_flat = q.reshape(B * Q, H, D).contiguous()
    k_flat = k.reshape(B * KV, H, D).contiguous()
    v_flat = v.reshape(B * KV, H, D).contiguous()
    kwargs = dict(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        softmax_scale=None,
        softcap=0,
    )

    out, _meta = flex_flash_attn_func(num_heads_q=H, num_heads_kv=H, head_dim=D, **kwargs)
    return out.reshape(B, Q, H, D)


@dataclass(frozen=True)
class BlockPattern:
    """
    Block schedule:
      - Number of tokens in a frame = frame_tokens (e.g., H_lat*W_lat)
      - Blocks are measured in frames:
          block0 has first_chunk_frames frames
          block1.. have chunk_frames frames each
        pattern [1 frame] [N frames] [N frames] ...
    """

    frame_tokens: int = 0
    first_chunk_frames: int = 1
    chunk_frames: int = 1  # N

    def get_block_tokens(self, block_idx: int) -> int:
        """Number of tokens in the given block index."""
        return self.first_chunk_frames * self.frame_tokens if block_idx == 0 else self.chunk_frames * self.frame_tokens

    def blocks_to_frames(self, num_blocks: int) -> int:
        """Total frames covered by the first `num_blocks` blocks starting from global block 0."""
        return 0 if num_blocks <= 0 else self.first_chunk_frames + (num_blocks - 1) * self.chunk_frames

    def blocks_to_tokens(self, num_blocks: int) -> int:
        """Total tokens covered by the first `num_blocks` blocks starting from global block 0."""
        return self.blocks_to_frames(num_blocks) * self.frame_tokens

    def block_size(self, block_idx: int) -> int:
        return self.first_chunk_frames if block_idx == 0 else self.chunk_frames

    def token_to_rel_block(self, token_idx, use_first_block: bool):
        # expect Tensor input
        frame_idx = token_idx // self.frame_tokens
        if use_first_block:
            return torch.where(
                frame_idx < self.first_chunk_frames,
                torch.zeros_like(frame_idx),
                1 + (frame_idx - self.first_chunk_frames) // self.chunk_frames,
            )
        else:
            return frame_idx // self.chunk_frames

    def spans(self, num_blocks: int, block_offset: int, count_tokens: bool = False) -> Tuple[List[Tuple[int, int]], int]:
        """
        Return block spans and total length.
        """
        spans: List[Tuple[int, int]] = []
        cur = 0

        multiplier = self.frame_tokens if count_tokens else 1

        for i in range(num_blocks):
            sz = (self.first_chunk_frames if (block_offset == 0 and i == 0) else self.chunk_frames) * multiplier
            spans.append((cur, cur + sz))
            cur += sz
        return spans, cur

    def block_bounds(self, q_bo: int, Q: int) -> torch.Tensor:
        """Block boundaries in token coords: [0, ..., Q]."""
        first_tok = self.first_chunk_frames * self.frame_tokens
        blk_tok = self.chunk_frames * self.frame_tokens

        bounds = [0]
        cur = 0
        if q_bo == 0:
            cur = min(Q, cur + first_tok)
            bounds.append(cur)
        while cur < Q:
            cur = min(Q, cur + blk_tok)
            bounds.append(cur)

        return bounds


@dataclass(frozen=True)
class AttnMaskSpec:
    """
    Attention mask spec (block-based):

    mode:
      - "none": full attention
      - "block_causal": blockwise causal on blocks: KV blocks <= Q block (bidirectional inside a block)
      - "teacher_forcing": reference-style teacher forcing on packed sequence [clean][noisy]
          * clean queries attend to clean blocks <= their block
          * noisy queries attend to (same noisy block) OR (previous clean blocks)
    """

    mode: Literal["none", "block_causal", "teacher_forcing"] = "none"

    pattern: BlockPattern = None

    # sliding window & attention sink
    local_attn_blocks: int = 0  # 0 disables; otherwise keep last K blocks (including current)
    sink_blocks: int = 0  # always allow attending to KV blocks < sink_blocks

    # offsets (in blocks) for cache/streaming (ignored for teacher_forcing)
    q_block_offset: int = 0

    # teacher forcing: number of clean blocks in the clean half
    clean_blocks: int = 0


def _ceil_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _cuda_compute_capability(device: torch.device) -> int:
    if torch.cuda.is_available() and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    return 0


def _flex_block_size(mask_block_size, device: torch.device) -> Tuple[int, int]:
    if isinstance(mask_block_size, tuple):
        return mask_block_size
    if os.environ.get("RCM_FLEX_BACKEND", "auto").lower() == "flash" and _cuda_compute_capability(device) >= 100:
        return 256, int(mask_block_size)
    return int(mask_block_size), int(mask_block_size)


class FlexOrSdpaLocalAttention(nn.Module):
    """
    local_attn(q,k,v): [B,S,H,D] -> [B,S,H,D]

    - mode="none": SDPA full attention
    - else: FlexAttention + BlockMask, with right-padding to multiples of mask_block_size (default 128)
    """

    def __init__(
        self,
        attn=attention,
        flex_attn=flex_attention_op,
        mask_block_size: int = 128,
        compile_block_mask: bool = True,
        backend: Optional[Literal["flex", "magi"]] = None,
    ):
        super().__init__()
        self.attn = attn
        self.flex_attn = flex_attn
        self.mask_block_size = mask_block_size
        self.compile_block_mask = compile_block_mask
        self.backend = (backend or os.environ.get("RCM_ATTENTION_BACKEND", "flex")).lower()
        if self.backend not in {"flex", "magi"}:
            raise ValueError(f"Unsupported attention backend {self.backend!r}; expected flex or magi")
        self._bm_cache: Dict[Tuple, object] = {}
        self._magi_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def clear_mask_cache(self):
        self._bm_cache.clear()
        self._magi_cache.clear()

    def _make_mask_fn(self, spec: AttnMaskSpec, Q_real: int, KV_real: int):
        mode = spec.mode

        if mode == "block_causal":
            pat = spec.pattern

            q_bo = spec.q_block_offset
            local_k = spec.local_attn_blocks - 1
            sink_k = spec.sink_blocks

            use_first_q = q_bo == 0

            def mask(b, h, q_idx, kv_idx):
                valid = (q_idx < Q_real) & (kv_idx < KV_real)

                q_rel_blk = pat.token_to_rel_block(q_idx, use_first_block=use_first_q)
                kv_rel_blk = pat.token_to_rel_block(kv_idx, use_first_block=True)
                q_blk = q_bo + q_rel_blk
                kv_blk = kv_rel_blk

                allow = kv_blk <= q_blk

                if local_k >= 0:
                    # keep last local_k blocks incl current
                    lo = q_blk - local_k
                    allow = allow & (kv_blk >= lo)

                if sink_k > 0:
                    allow = allow | ((kv_blk < sink_k) & (kv_blk <= q_blk))

                return valid & allow

            sig = ("block_causal", pat, q_bo, local_k, sink_k, Q_real, KV_real)
            return mask, sig

        if mode == "teacher_forcing":
            pat = spec.pattern

            clean_blocks = spec.clean_blocks
            local_k = spec.local_attn_blocks - 1
            sink_k = spec.sink_blocks

            assert clean_blocks > 0, "clean_blocks must be set for teacher_forcing"

            clean_len = pat.blocks_to_tokens(clean_blocks)
            total_len = 2 * clean_len

            def mask(b, h, q_idx, kv_idx):
                valid = (q_idx < Q_real) & (kv_idx < KV_real) & (q_idx < total_len) & (kv_idx < total_len)

                eye = q_idx == kv_idx

                q_in_clean = q_idx < clean_len
                kv_in_clean = kv_idx < clean_len

                # block ids within half
                q_blk_clean = pat.token_to_rel_block(q_idx, use_first_block=True)
                q_blk_noisy = pat.token_to_rel_block(q_idx - clean_len, use_first_block=True)

                kv_blk_clean = pat.token_to_rel_block(kv_idx, use_first_block=True)
                kv_blk_noisy = pat.token_to_rel_block(kv_idx - clean_len, use_first_block=True)

                # ---- clean queries ----
                clean_allow = q_in_clean & kv_in_clean & (kv_blk_clean <= q_blk_clean)
                if local_k >= 0:
                    lo = q_blk_clean - local_k
                    clean_allow = clean_allow & (kv_blk_clean >= lo)
                clean_allow = clean_allow | (q_in_clean & eye)

                # ---- noisy queries ----
                # (a) attend to previous clean blocks (strictly < q_blk)
                noisy_to_clean = (~q_in_clean) & kv_in_clean & (kv_blk_clean < q_blk_noisy)
                if local_k >= 0:
                    lo = q_blk_noisy - local_k
                    noisy_to_clean = noisy_to_clean & (kv_blk_clean >= lo)

                # (b) attend to same noisy block
                noisy_to_noisy = (~q_in_clean) & (~kv_in_clean) & (kv_blk_noisy == q_blk_noisy)

                noisy_allow = noisy_to_clean | noisy_to_noisy | ((~q_in_clean) & eye)

                # sink blocks
                if sink_k > 0:
                    noisy_allow = noisy_allow | ((~q_in_clean) & kv_in_clean & (kv_blk_clean < sink_k) & (kv_blk_clean < q_blk_noisy))
                    clean_allow = clean_allow | (q_in_clean & kv_in_clean & (kv_blk_clean < sink_k) & (kv_blk_clean < q_blk_clean))

                return valid & (eye | clean_allow | noisy_allow)

            sig = ("teacher_forcing", pat, clean_blocks, local_k, sink_k, Q_real, KV_real)
            return mask, sig

        raise ValueError(f"Unknown mask mode: {mode}")

    def _get_block_mask(self, spec: AttnMaskSpec, Q_real: int, KV_real: int, Q_pad: int, KV_pad: int, device: torch.device, block_size):
        mask_fn, sig = self._make_mask_fn(spec, Q_real=Q_real, KV_real=KV_real)
        key = (sig, Q_pad, KV_pad, device.type, device.index, block_size, self.compile_block_mask)

        bm = self._bm_cache.get(key, None)
        if bm is None:
            bm = create_block_mask(
                mask_fn,
                B=None,
                H=None,
                Q_LEN=Q_pad,
                KV_LEN=KV_pad,
                device=device,
                BLOCK_SIZE=block_size,
                _compile=self.compile_block_mask,
            )
            self._bm_cache[key] = bm
        return bm

    def _get_magi_ranges(self, spec: AttnMaskSpec, Q_real: int, KV_real: int, device: torch.device):
        key = (spec, Q_real, KV_real, device.type, device.index)
        cached = self._magi_cache.get(key, None)
        if cached is None:
            from rcm.utils.magimask import make_magi_ranges_full

            cached = make_magi_ranges_full(spec, Q_real=Q_real, KV_real=KV_real, device=device)
            self._magi_cache[key] = cached
        return cached

    def forward(self, q, k, v, attn_meta: Optional[AttnMaskSpec] = None, **_ignored):
        # q,k,v: [B, S, H, D]
        B, Q, H, D = q.shape
        KV = k.shape[1]
        assert k.shape == (B, KV, H, D) and v.shape == (B, KV, H, D)

        spec = attn_meta if attn_meta is not None else AttnMaskSpec(mode="none")

        # Fast path: SDPA full attention
        if spec.mode == "none":
            return self.attn(q, k, v)

        if spec.mode == "block_causal" and spec.local_attn_blocks == 0:
            pat = spec.pattern
            q_blk = spec.q_block_offset

            if (Q == pat.get_block_tokens(q_blk)) and (KV <= pat.blocks_to_tokens(q_blk + 1)):
                return self.attn(q, k, v)

        if self.backend == "magi":
            q_ranges, k_ranges = self._get_magi_ranges(spec, Q_real=Q, KV_real=KV, device=q.device)
            return _call_magi_flex_flash(q, k, v, q_ranges, k_ranges)

        q_block_size, kv_block_size = _flex_block_size(self.mask_block_size, q.device)
        Q_pad = _ceil_to_multiple(Q, q_block_size)
        KV_pad = _ceil_to_multiple(KV, kv_block_size)

        block_mask = self._get_block_mask(
            spec, Q_real=Q, KV_real=KV, Q_pad=Q_pad, KV_pad=KV_pad, device=q.device, block_size=(q_block_size, kv_block_size)
        )

        # right-pad q to Q_pad
        if Q_pad != Q:
            q_pad = torch.cat([q, q.new_zeros((B, Q_pad - Q, H, D))], dim=1)
        else:
            q_pad = q

        # right-pad kv to KV_pad
        if KV_pad != KV:
            k_pad = torch.cat([k, k.new_zeros((B, KV_pad - KV, H, D))], dim=1)
            v_pad = torch.cat([v, v.new_zeros((B, KV_pad - KV, H, D))], dim=1)
        else:
            k_pad, v_pad = k, v

        return self.flex_attn(q_pad, k_pad, v_pad, block_mask)[:, :Q]


def visualize_attn_mask_spec(
    local_attn,
    spec,
    q_blocks: Optional[int] = None,
    kv_blocks: Optional[int] = None,
    figsize=(7, 7),
    ax: Optional[object] = None,
    title: Optional[str] = None,
    show_block_grid: bool = True,
):
    import matplotlib.pyplot as plt

    specv = replace(spec, pattern=replace(spec.pattern, frame_tokens=1))

    if specv.mode == "teacher_forcing":
        if specv.clean_blocks <= 0:
            raise ValueError("teacher_forcing requires spec.clean_blocks > 0")

        clean_spans, clean_len = specv.pattern.spans(num_blocks=specv.clean_blocks, block_offset=0)
        Q = K = 2 * clean_len
        mask_fn, _ = local_attn._make_mask_fn(specv, Q_real=Q, KV_real=K)

        spans = clean_spans + [(s + clean_len, e + clean_len) for (s, e) in clean_spans]
        labels = [f"c{i}" for i in range(specv.clean_blocks)] + [f"n{i}" for i in range(specv.clean_blocks)]
        half_cut = clean_len

        q_spans = kv_spans = spans

    else:
        if q_blocks is None:
            raise ValueError("block_causal visualization requires q_blocks")
        if kv_blocks is None:
            kv_blocks = q_blocks

        q_spans, Q = specv.pattern.spans(num_blocks=q_blocks, block_offset=specv.q_block_offset)
        kv_spans, K = specv.pattern.spans(num_blocks=kv_blocks, block_offset=0)
        mask_fn, _ = local_attn._make_mask_fn(specv, Q_real=Q, KV_real=K)

        q0 = specv.q_block_offset
        labels_y = [str(q0 + i) for i in range(len(q_spans))]
        labels_x = [str(i) for i in range(len(kv_spans))]
        half_cut = None

    mask = np.fromiter(
        (1 if mask_fn(0, 0, torch.tensor(i), torch.tensor(j)) else 0 for i in range(Q) for j in range(K)),
        dtype=np.uint8,
        count=Q * K,
    ).reshape(Q, K)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(mask, interpolation="nearest", aspect="auto", origin="upper")
    ax.set_xlabel("KV (frames)")
    ax.set_ylabel("Q (frames)")
    ax.set_title(title or f"Attn mask (frame-level) | mode={spec.mode}")

    if show_block_grid:
        for s, _e in q_spans:
            if s != 0:
                ax.axhline(s - 0.5, linewidth=0.8, alpha=0.35)
        for s, _e in kv_spans:
            if s != 0:
                ax.axvline(s - 0.5, linewidth=0.8, alpha=0.35)
        if half_cut is not None:
            ax.axhline(half_cut - 0.5, linewidth=1.2, alpha=0.6)
            ax.axvline(half_cut - 0.5, linewidth=1.2, alpha=0.6)

    def _centers(spans):
        return [(s + e - 1) / 2.0 for (s, e) in spans]

    if specv.mode == "teacher_forcing":
        centers = _centers(kv_spans)
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(_centers(q_spans))
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks(_centers(kv_spans))
        ax.set_xticklabels(labels_x, rotation=90)
        ax.set_yticks(_centers(q_spans))
        ax.set_yticklabels(labels_y)

    fig.tight_layout()
    return fig, ax, mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Visualization examples ---
    local_attn = FlexOrSdpaLocalAttention()
    spec = AttnMaskSpec(
        mode="block_causal",
        pattern=BlockPattern(frame_tokens=1560, first_chunk_frames=1, chunk_frames=4),
        q_block_offset=2,
        local_attn_blocks=3,
        sink_blocks=1,
    )
    fig, ax, mask = visualize_attn_mask_spec(local_attn, spec, q_blocks=4, kv_blocks=6, title="Block causal")
    plt.savefig("block_causal_example.png")

    tf_spec = AttnMaskSpec(
        mode="teacher_forcing",
        pattern=BlockPattern(frame_tokens=1560, first_chunk_frames=1, chunk_frames=4),
        clean_blocks=6,
        local_attn_blocks=3,
        sink_blocks=1,
    )
    fig, ax, mask = visualize_attn_mask_spec(local_attn, tf_spec, title="Teacher forcing")
    plt.savefig("teacher_forcing_example.png")
