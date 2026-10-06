# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Any, Callable, Tuple, Union, Dict

import torch
import torch.distributed as dist
from torch import Tensor, nn

from rcm.utils.a2a_cp import async_a2a_communicate
from rcm.utils.flash_attention_jvp_triton import _attention
from rcm.utils.blockmask import AttnMaskSpec
from rcm.utils.magimask import MagiMask, make_magi_ranges_full, prepare_magi_csr_and_tasks
from rcm.utils.attention import get_device_cc
from rcm.utils.kv_cache import AttnContext, KVCacheMode
from rcm.utils.rope import apply_rope_with_tangent

TensorWithT = Tuple[torch.Tensor, torch.Tensor]


class JVP(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        withT = kwargs.pop("withT", False)
        if withT:
            return self._forward_jvp(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward_jvp(self, *args, **kwargs):
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        raise NotImplementedError


def torch_attention_op_withT(q_B_S_H_D_withT: TensorWithT, k_B_S_H_D_withT: TensorWithT, v_B_S_H_D_withT: TensorWithT, mask: MagiMask = None):
    q_B_S_H_D, t_q_B_S_H_D = q_B_S_H_D_withT
    k_B_S_H_D, t_k_B_S_H_D = k_B_S_H_D_withT
    v_B_S_H_D, t_v_B_S_H_D = v_B_S_H_D_withT
    result_B_H_S_D, t_result_B_H_S_D = _attention.apply(
        q_B_S_H_D.transpose(1, 2),
        k_B_S_H_D.transpose(1, 2),
        v_B_S_H_D.transpose(1, 2),
        t_q_B_S_H_D.transpose(1, 2),
        t_k_B_S_H_D.transpose(1, 2),
        t_v_B_S_H_D.transpose(1, 2),
        None,
        mask,
    )
    return (result_B_H_S_D.transpose(1, 2), t_result_B_H_S_D.transpose(1, 2).detach())


class _SeqAllToAllQKVWithT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        q: Tensor,
        t_q: Tensor,
        k: Tensor,
        t_k: Tensor,
        v: Tensor,
        t_v: Tensor,
        cp_size: int,
        cp_stream: torch.cuda.Stream,
        local_seq_2_local_head: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_stream = cp_stream
        ctx.local_seq_2_local_head = local_seq_2_local_head
        q, t_q, k, t_k, v, t_v = async_a2a_communicate([q, t_q, k, t_k, v, t_v], cp_size, group, cp_stream, local_seq_2_local_head)
        return q, t_q, k, t_k, v, t_v

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, None, None, None]:
        q_grad, t_q_grad, k_grad, t_k_grad, v_grad, t_v_grad = _SeqAllToAllQKVWithT.apply(
            ctx.group, *grad_output, ctx.cp_size, ctx.cp_stream, not ctx.local_seq_2_local_head
        )
        return (None, q_grad, t_q_grad, k_grad, t_k_grad, v_grad, t_v_grad, None, None, None)


class _SeqAllToAllOutputWithT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        output,
        t_output,
        cp_size: int,
        cp_stream: torch.cuda.Stream,
        local_seq_2_local_head: bool,
    ) -> Tuple[Tensor, Tensor]:
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_stream = cp_stream
        ctx.local_seq_2_local_head = local_seq_2_local_head
        output, t_output = async_a2a_communicate([output, t_output], cp_size, group, cp_stream, local_seq_2_local_head)
        return output, t_output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, Tensor, None, None, None]:
        output_grad, t_output_grad = _SeqAllToAllOutputWithT.apply(
            ctx.group, *grad_output, ctx.cp_size, ctx.cp_stream, not ctx.local_seq_2_local_head
        )
        return (None, output_grad, t_output_grad, None, None, None)


class MinimalA2AAttnOpWithT(torch.nn.Module):
    def __init__(self, local_attn_T: Union[nn.Module, Callable] = torch_attention_op_withT):
        super().__init__()
        self.local_attn_T = local_attn_T
        self.pg = None
        self.stream = None

    def _materialize_kv(self, key_withT, value_withT, attn_ctx):
        # cached K/V have zero tangents.
        if attn_ctx is None or attn_ctx.kv_cache is None:
            return key_withT, value_withT

        k, tk = key_withT
        v, tv = value_withT
        cache = attn_ctx.kv_cache

        if attn_ctx.mode == KVCacheMode.DISABLED:
            return key_withT, value_withT
        if attn_ctx.mode == KVCacheMode.APPEND:
            assert attn_ctx.block_range == len(cache._cum_ends)
            cache.append(k, v)
            k_all, v_all = cache.get()
            tk_all = torch.zeros_like(k_all)
            tv_all = torch.zeros_like(v_all)
            return (k_all, tk_all), (v_all, tv_all)
        if attn_ctx.mode == KVCacheMode.READONLY:
            k_cached, v_cached = cache.get(attn_ctx.block_range)
            if k_cached is None:
                return key_withT, value_withT
            tk_cached = torch.zeros_like(k_cached)
            tv_cached = torch.zeros_like(v_cached)
            return (
                (torch.cat([k_cached, k], dim=1), torch.cat([tk_cached, tk], dim=1)),
                (torch.cat([v_cached, v], dim=1), torch.cat([tv_cached, tv], dim=1)),
            )

        raise ValueError(f"Unknown KVCacheMode: {attn_ctx.mode}")

    def forward(
        self, query_withT: TensorWithT, key_withT: TensorWithT, value_withT: TensorWithT, *args: Any, attn_ctx: Optional[AttnContext] = None, **kwargs
    ) -> TensorWithT:
        pg_size = 1 if self.pg is None else dist.get_world_size(self.pg)

        if pg_size > 1:
            query_B_S_H_D, t_query_B_S_H_D = query_withT
            key_B_S_H_D, t_key_B_S_H_D = key_withT
            value_B_S_H_D, t_value_B_S_H_D = value_withT

            (query_B_S_H_D, t_query_B_S_H_D, key_B_S_H_D, t_key_B_S_H_D, value_B_S_H_D, t_value_B_S_H_D) = _SeqAllToAllQKVWithT.apply(
                self.pg, query_B_S_H_D, t_query_B_S_H_D, key_B_S_H_D, t_key_B_S_H_D, value_B_S_H_D, t_value_B_S_H_D, pg_size, self.stream, True
            )
            query_withT = (query_B_S_H_D, t_query_B_S_H_D)
            key_withT = (key_B_S_H_D, t_key_B_S_H_D)
            value_withT = (value_B_S_H_D, t_value_B_S_H_D)

        rope = attn_ctx.rope if attn_ctx is not None else None

        if rope is not None and rope.cached_k_rotated and rope.current_key_freqs is not None:
            k, tk = key_withT
            k, tk = apply_rope_with_tangent(k, tk, rope.current_key_freqs, fused=rope.use_fused)
            key_withT = (k, tk.detach())

        key_withT, value_withT = self._materialize_kv(key_withT, value_withT, attn_ctx)

        if rope is not None and rope.query_freqs is not None:
            q, tq = query_withT
            q, tq = apply_rope_with_tangent(q, tq, rope.query_freqs, fused=rope.use_fused)
            query_withT = (q, tq.detach())
            if not rope.cached_k_rotated:
                k, tk = key_withT
                k, tk = apply_rope_with_tangent(k, tk, rope.key_freqs, fused=rope.use_fused)
                key_withT = (k, tk.detach())

        output_withT = self.local_attn_T(query_withT, key_withT, value_withT, *args, **kwargs)

        if pg_size > 1:
            output_B_S_H_D, t_output_B_S_H_D = output_withT
            output_B_S_H_D, t_output_B_S_H_D = _SeqAllToAllOutputWithT.apply(self.pg, output_B_S_H_D, t_output_B_S_H_D, pg_size, self.stream, False)
            output_withT = (output_B_S_H_D, t_output_B_S_H_D)

        return output_withT

    def set_context_parallel_group(self, process_group, ranks, stream):
        del ranks
        self.pg = process_group
        self.stream = stream


class FlexOrSdpaLocalAttentionWithT(nn.Module):
    def __init__(self, attn_T=torch_attention_op_withT):
        super().__init__()
        self.attn_T = attn_T
        self._cache: Dict[Tuple, MagiMask] = {}

    def get_magi_mask(self, spec: AttnMaskSpec, Q_real: int, KV_real: int, device, BLOCK_M: int) -> MagiMask:
        key = (
            spec.mode,
            spec.pattern,
            int(spec.local_attn_blocks),
            int(spec.sink_blocks),
            int(spec.q_block_offset),
            int(spec.clean_blocks),
            int(Q_real),
            int(KV_real),
            int(BLOCK_M),
            device.type,
            device.index,
        )
        mask = self._cache.get(key, None)
        if mask is None:
            q_ranges, k_ranges = make_magi_ranges_full(spec, Q_real, KV_real, device)
            mask = prepare_magi_csr_and_tasks(q_ranges, k_ranges, Q_real, KV_real, BLOCK_M)
            self._cache[key] = mask
        return mask

    def clear_cache(self):
        self._cache.clear()

    def forward(self, q_withT, k_withT, v_withT, attn_meta: Optional[AttnMaskSpec] = None, **_ignored):
        # q,k,v: [B, S, H, D]
        q, _ = q_withT
        k, _ = k_withT
        v, _ = v_withT
        B, Q, H, D = q.shape
        KV = k.shape[1]
        assert k.shape == (B, KV, H, D) and v.shape == (B, KV, H, D)

        spec = attn_meta if attn_meta is not None else AttnMaskSpec(mode="none")

        # Fast path: SDPA full attention
        if spec.mode == "none":
            return self.attn_T(q_withT, k_withT, v_withT)

        if spec.mode == "block_causal" and int(spec.local_attn_blocks) == 0:
            pat = spec.pattern
            q_blk = int(spec.q_block_offset)

            if (Q == pat.get_block_tokens(q_blk)) and (KV <= pat.blocks_to_tokens(q_blk + 1)):
                return self.attn_T(q_withT, k_withT, v_withT)

        BLOCK_M = 64 if get_device_cc(q.device) == 90 else 128

        mask = self.get_magi_mask(spec, Q_real=Q, KV_real=KV, device=q.device, BLOCK_M=BLOCK_M)

        return self.attn_T(q_withT, k_withT, v_withT, mask)


def naive_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1).to(query.dtype)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def naive_attention_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D):
    return naive_scaled_dot_product_attention(q_B_S_H_D.transpose(1, 2), k_B_S_H_D.transpose(1, 2), v_B_S_H_D.transpose(1, 2)).transpose(1, 2)


def torch_attention_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D):
    return torch.nn.functional.scaled_dot_product_attention(
        q_B_S_H_D.transpose(1, 2), k_B_S_H_D.transpose(1, 2), v_B_S_H_D.transpose(1, 2)
    ).transpose(1, 2)
