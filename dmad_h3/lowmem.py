"""Low-VRAM path: the 32B text encoder and the 33B transformer are streamed to the GPU one layer at a time.

The numerics are those of the resident model: the same Diffusers / Transformers modules run with the same dtypes
and the same op order, only *where the weights live* differs.

* A streamed layer's weights are read from the safetensors shards (`cache_in_ram=False`: no host memory beyond the
  page cache) or from a pinned host-memory copy made once (`cache_in_ram=True`), copied into one of two rotating GPU
  buffers on a side stream while the previous layer computes, and bound to the layer's parameters right before its
  forward.
* The transformer's AdaLN projections (13B of its 33B parameters) depend only on the sampling step, so their outputs
  are computed once per step up front and the projections are replaced by table lookups; the per-step stream is then
  0.69 GB per block (35 GB per model evaluation) instead of 1.3 GB.
* Everything else stays resident: token refiner, embedders, output heads, time embedder, the LoRA.
* Optionally (`chunk_rows`) the transformer blocks run row-chunked for long videos, see `_chunked_block_forward`.
"""

import json
import math
import queue
import sys
import threading
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

_ALIGN = 256
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _safe_open(path):
    """On Windows a memory-mapped shard is charged in full to the system commit for as long as it stays open, so
    shards are read with positioned reads there (safetensors >= 0.8.0); other platforms keep the default mmap."""
    if sys.platform == "win32":
        return safe_open(str(path), "pt", backend="pread")
    return safe_open(str(path), "pt")


class ShardIndex:
    """tensor name -> shard of a safetensors directory; shards stay open once touched."""

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.where = {}
        index = sorted(self.directory.glob("*.safetensors.index.json"))
        if index:
            for key, fname in json.load(open(index[0]))["weight_map"].items():
                self.where[key] = self.directory / fname
        else:
            for path in sorted(self.directory.glob("*.safetensors")):
                with _safe_open(path) as f:
                    for key in f.keys():
                        self.where[key] = path
        if not self.where:
            raise FileNotFoundError(f"no safetensors in {self.directory}")
        self._handles = {}
        self._lock = threading.Lock()

    def _handle(self, path):
        with self._lock:
            if path not in self._handles:
                self._handles[path] = _safe_open(path)
            return self._handles[path]

    def keys(self):
        return self.where.keys()

    def info(self, key):
        """(shape, dtype) without reading the data."""
        sl = self._handle(self.where[key]).get_slice(key)
        return tuple(sl.get_shape()), _DTYPES[sl.get_dtype()]

    def tensor(self, key):
        return self._handle(self.where[key]).get_tensor(key)


def _find_param(root, key):
    """(module, attribute) holding parameter `key`; also finds it when PEFT wrapped the linear (`.base_layer.`)."""
    head, _, attr = key.rpartition(".")
    for module_path in (head, head + ".base_layer"):
        try:
            module = root.get_submodule(module_path) if module_path else root
        except AttributeError:
            continue
        if attr in module._parameters:
            return module, attr
    raise KeyError(f"{key} is not a parameter of the model")


def _bind(module, attr, tensor):
    module._parameters[attr] = nn.Parameter(tensor, requires_grad=False)


def _unbind(module, attr):
    p = module._parameters[attr]
    module._parameters[attr] = nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=False)


def load_resident(model, source, keys, device):
    """Materialize `keys` on `device` in the dtype the skeleton gave them."""
    for key in keys:
        module, attr = _find_param(model, key)
        _bind(module, attr, source.tensor(key).to(device=device, dtype=module._parameters[attr].dtype))


def _buffers_to(model, device):
    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.device.type != "meta":
                module._buffers[name] = buf.to(device)


def _aligned(nbytes):
    return -(-nbytes // _ALIGN) * _ALIGN


# int8 weight codec: per-output-row symmetric scales, `w ≈ q * s[:, None]`; dequantized to bf16 on the GPU right
# before use, so the GEMMs themselves are unchanged. Halves the host memory and the per-step transfer.
_CODECS = {"int8": (torch.int8, 127.0)}


def _quantize(w, codec):
    qdtype, qmax = _CODECS[codec]
    w32 = w.float()
    scales = w32.abs().amax(dim=1) / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    q = (w32 / scales[:, None]).round_()
    return q.clamp_(-qmax, qmax).to(qdtype), scales


class _Item:
    __slots__ = ("module", "attr", "key", "shape", "dtype", "nbytes", "quant", "offset", "raw_offset")

    def __init__(self, module, attr, key, shape, dtype, nbytes, quant):
        self.module, self.attr, self.key, self.shape, self.dtype, self.nbytes, self.quant = module, attr, key, shape, dtype, nbytes, quant


class _Group:
    """The parameters of one streamed layer, laid out back to back in a flat byte buffer (the *stored* layout:
    bf16, or 8-bit data followed by fp32 row scales for the quantized matrices)."""

    def __init__(self, model, source, keys, codec=None):
        self.codec = codec
        self.items, offset, raw_offset = [], 0, 0
        for key in keys:
            module, attr = _find_param(model, key)
            shape, dtype = source.info(key)
            if module._parameters[attr].dtype != dtype:
                raise ValueError(f"{key}: checkpoint {dtype} but model {module._parameters[attr].dtype}; cannot stream")
            numel = math.prod(shape)
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            quant = codec is not None and len(shape) == 2 and numel >= 2**20
            item = _Item(module, attr, key, shape, dtype, nbytes, quant)
            item.offset, item.raw_offset = offset, raw_offset
            self.items.append(item)
            offset += _aligned(numel + shape[0] * 4 if quant else nbytes)
            raw_offset += _aligned(nbytes)
        self.nbytes = offset  # stored layout (what is transferred)
        self.raw_bytes = raw_offset  # bf16 layout (what the layer computes with)

    def stored(self, flat, item):
        """The views of `item` inside a stored-layout buffer: (tensor, None) or (8-bit data, fp32 scales)."""
        if not item.quant:
            return flat[item.offset : item.offset + item.nbytes].view(item.dtype).view(item.shape), None
        numel = math.prod(item.shape)
        data = flat[item.offset : item.offset + numel].view(_CODECS[self.codec][0]).view(item.shape)
        scales = flat[item.offset + numel : item.offset + numel + item.shape[0] * 4].view(torch.float32)
        return data, scales

    def raw(self, scratch, item):
        return scratch[item.raw_offset : item.raw_offset + item.nbytes].view(item.dtype).view(item.shape)


class Streamer:
    """Streams groups of parameters through two rotating GPU buffers, one group ahead of the compute.

    Group i is bound to its modules by a forward pre-hook on `modules[i]` and unbound by the forward hook, which also
    requests group i + 2 (wrapping around, so the next forward pass finds its first groups already in flight). A
    worker thread fills the free buffer: from the shards through a pinned staging buffer, or straight from the pinned
    host copies when `cache_in_ram`.
    """

    def __init__(self, model, source, key_groups, modules, device, cache_in_ram=False, codec=None, cache_dir=None, log=None):
        if len(key_groups) != len(modules) or len(modules) < 2:
            raise ValueError("one key group per streamed module, at least two")
        if codec is not None and not cache_in_ram and cache_dir is None:
            raise ValueError("int8 weights streamed from disk need a cache_dir for the quantized copy")
        self.groups = [_Group(model, source, keys, codec) for keys in key_groups]
        self.source, self.device, self.cache_in_ram, self.codec = source, device, cache_in_ram, codec
        slot_bytes = max(g.nbytes for g in self.groups)
        self.slots = [torch.empty(slot_bytes, dtype=torch.uint8, device=device) for _ in range(2)]
        self.scratch = torch.empty(max(g.raw_bytes for g in self.groups), dtype=torch.uint8, device=device) if codec else None
        self.stream = torch.cuda.Stream(device=device)
        self.compute_done = [torch.cuda.Event() for _ in range(2)]
        self.copy_done = [None, None]
        self.ready, self.cond, self.requests = {}, threading.Condition(), queue.Queue()
        self.cache_files = None
        if cache_in_ram:
            self.host = []
            for i, g in enumerate(self.groups):
                self.host.append(self._read(g, torch.empty(g.nbytes, dtype=torch.uint8).pin_memory()))
                if log and (i + 1) % 10 == 0:
                    log(f"cached {i + 1}/{len(self.groups)} layers in host memory")
        else:
            self.staging = [torch.empty(slot_bytes, dtype=torch.uint8).pin_memory() for _ in range(2)]
            if codec is not None:
                self.cache_files = self._quantized_cache(Path(cache_dir).expanduser(), log)
        self.hooks = []
        for i, module in enumerate(modules):
            self.hooks.append(module.register_forward_pre_hook(self._pre(i), with_kwargs=True))
            self.hooks.append(module.register_forward_hook(self._post(i)))
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        self.requests.put(0)
        self.requests.put(1)
        self.bytes_per_pass = sum(g.nbytes for g in self.groups)

    def _quantized_cache(self, cache_dir, log):
        """One file per group in the stored layout, written once (quantizing on the GPU) and validated against a
        manifest of the shards it was built from."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        shards = sorted({str(p) for p in self.source.where.values()})
        manifest = {"codec": self.codec, "groups": [g.nbytes for g in self.groups],
                    "shards": [[Path(p).name, Path(p).stat().st_size] for p in shards]}
        files = [cache_dir / f"group_{i:03d}.bin" for i in range(len(self.groups))]
        mpath = cache_dir / "manifest.json"
        valid = (mpath.is_file() and json.load(open(mpath)) == manifest
                 and all(f.is_file() and f.stat().st_size == g.nbytes for f, g in zip(files, self.groups)))
        if not valid:
            if log:
                log(f"writing the {self.codec} copy of the streamed weights to {cache_dir} (once)")
            for i, (f, g) in enumerate(zip(files, self.groups)):
                flat = self._read(g, torch.empty(g.nbytes, dtype=torch.uint8))
                with open(f, "wb") as h:
                    h.write(memoryview(flat.numpy()))
                if log and (i + 1) % 10 == 0:
                    log(f"  {i + 1}/{len(self.groups)} layers written")
            json.dump(manifest, open(mpath, "w"))
        return files

    def _read_cached(self, g, flat):
        n = self.groups[g].nbytes
        with open(self.cache_files[g], "rb") as h:
            h.readinto(memoryview(flat.numpy())[:n])
        return flat

    def _read(self, group, flat):
        """Fill a stored-layout host buffer from the shards (quantizing on the GPU when a codec is set)."""
        for item in group.items:
            data, scales = group.stored(flat, item)
            if item.quant:
                q, s = _quantize(self.source.tensor(item.key).to(self.device), self.codec)
                data.copy_(q)
                scales.copy_(s)
            else:
                data.copy_(self.source.tensor(item.key))
        return flat

    def _run(self):
        torch.cuda.set_device(self.device)
        try:
            while True:
                g = self.requests.get()
                if g is None:
                    return
                slot, group = g % 2, self.groups[g]
                if self.cache_in_ram:
                    host = self.host[g]
                else:
                    if self.copy_done[slot] is not None:
                        self.copy_done[slot].synchronize()  # the previous copy out of this staging buffer is done
                    if self.cache_files is not None:
                        host = self._read_cached(g, self.staging[slot])
                    else:
                        host = self._read(group, self.staging[slot])
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(self.compute_done[slot])  # the GPU is done with the buffer's previous occupant
                    self.slots[slot][: group.nbytes].copy_(host[: group.nbytes], non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self.stream)
                self.copy_done[slot] = event
                with self.cond:
                    self.ready[g] = event
                    self.cond.notify_all()
        except Exception as e:  # surface in the compute thread instead of hanging it
            with self.cond:
                self.error = e
                self.cond.notify_all()

    error = None

    def _pre(self, i):
        def hook(module, args, kwargs):
            with self.cond:
                while i not in self.ready and self.error is None:
                    self.cond.wait()
                if self.error is not None:
                    raise RuntimeError("weight streaming failed") from self.error
                event = self.ready.pop(i)
            torch.cuda.current_stream().wait_event(event)
            group, slot = self.groups[i], self.slots[i % 2]
            for item in group.items:
                data, scales = group.stored(slot, item)
                if item.quant:  # dequantize into the bf16 scratch buffer on the compute stream
                    weight = group.raw(self.scratch, item)
                    weight.copy_(data.float().mul_(scales[:, None]))
                    _bind(item.module, item.attr, weight)
                else:
                    _bind(item.module, item.attr, data)

        return hook

    def _post(self, i):
        def hook(module, args, output):
            self.compute_done[i % 2].record(torch.cuda.current_stream())
            for item in self.groups[i].items:
                _unbind(item.module, item.attr)
            self.requests.put((i + 2) % len(self.groups))

        return hook

    def close(self):
        for h in self.hooks:
            h.remove()
        self.requests.put(None)
        self.worker.join()
        torch.cuda.synchronize(self.device)
        self.slots = self.scratch = self.host = self.staging = None


# ----------------------------------------------------------------------------------------------------------------
# text encoder


class LowMemTextEncoder:
    """The H3 conditioner (Qwen3-VL-32B) with only the embedding table resident; decoder layers are streamed.

    Only the first `num_layers` decoder layers run (the conditioning is hidden layer `num_layers`), the vision tower
    and the LM head are never materialized.
    """

    def __init__(self, model_dir, device, num_layers, dtype=torch.bfloat16, cache_in_ram=False, log=None):
        from accelerate import init_empty_weights
        from transformers import AutoConfig, Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        root = Path(model_dir).expanduser().resolve()
        self.device, self.dtype = device, dtype
        config = AutoConfig.from_pretrained(str(root / "text_encoder"), local_files_only=True)
        with init_empty_weights(include_buffers=False):
            try:
                self.model = Qwen3VLForConditionalGeneration._from_config(config, dtype=dtype)
            except TypeError:
                self.model = Qwen3VLForConditionalGeneration._from_config(config, torch_dtype=dtype)
        self.model.eval().requires_grad_(False)
        lm = self.model.model.language_model
        lm.layers = lm.layers[:num_layers]
        _buffers_to(self.model, device)
        self.source = ShardIndex(root / "text_encoder")
        prefix = "model.language_model."
        load_resident(self.model, self.source, [prefix + "embed_tokens.weight", prefix + "norm.weight"], device)
        groups = [[k for k in self.source.keys() if k.startswith(f"{prefix}layers.{i}.")] for i in range(num_layers)]
        self.streamer = Streamer(self.model, self.source, groups, list(lm.layers), device, cache_in_ram, log=log)
        self._captured = None
        lm.layers[-1].register_forward_hook(self._capture)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(str(root / "tokenizer"), local_files_only=True)
        self.processor = Qwen3VLProcessor.from_pretrained(str(root / "processor"), local_files_only=True)

    def _capture(self, module, args, output):
        self._captured = output[0] if isinstance(output, tuple) else output

    @torch.inference_mode()
    def encode(self, prompt, output_dtype=torch.bfloat16):
        from .text_encoder import TEXT_TAG, prompt_inputs

        input_ids, mm_token_type_ids = prompt_inputs(self.tokenizer, self.processor, prompt, self.device)
        self.model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=None,
            image_grid_thw=None,
            use_cache=False,
        )
        prompt_embeds = self._captured.to(dtype=output_dtype).cpu()
        self._captured = None
        return {"prompt_embeds": prompt_embeds, "text_token_tags": torch.full((input_ids.shape[1],), TEXT_TAG, dtype=torch.long)}

    def close(self):
        self.streamer.close()
        self.model = self.source = None
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------
# transformer


class AdaLNTables(nn.Module):
    """Stands in for a block's `adaln_proj`: returns the six modulation tensors precomputed for the step whose
    timestep embedding is being passed (matched exactly against the embeddings the tables were built from)."""

    def __init__(self, tembs, tables):
        super().__init__()
        self.tembs = tembs  # per step: [num_timesteps, time_embed_dim] (one distinct noise level at the first step, two later)
        self.tables = tables  # per step: [6, num_timesteps * 3, hidden]

    def forward(self, temb):
        for step, known in enumerate(self.tembs):
            if known.shape == temb.shape and torch.equal(known, temb):
                return tuple(self.tables[step])
        raise RuntimeError("timestep not among the precomputed AdaLN tables; the sampling schedule changed")


def _chunked_block_forward(block, rows):
    """A `MiniMaxH3TransformerBlock.forward` that holds at most `rows` rows of every intermediate except q, k, v and
    the attention output. Everything but the attention itself is per row (norms, AdaLN modulation, projections,
    rotary, feed-forward), so those run chunk by chunk and the residual stream is updated in place; the attention
    still sees the whole sequence. Same ops and dtypes as the stock forward, but the matmuls run on smaller matrices,
    so results can differ from it in the last bits. Before q, k, v are allocated the allocator's free cached segments
    are released, so allocators without `expandable_segments` (Windows) find the room unfragmented."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

    attn = block.attn

    def forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(temb)
        cos, sin = rotary_emb
        n = hidden_states.shape[1]
        size = -(-n // -(-n // rows))  # equal chunks of at most `rows` (no small tail chunk with its own matmul kernel)
        spans = [(s, min(s + size, n)) for s in range(0, n, size)]
        torch.cuda.empty_cache()
        q, k, v = (hidden_states.new_empty(hidden_states.shape[0], n, attn.heads, attn.head_dim) for _ in range(3))
        for s, e in spans:
            idx = adaln_indices[s:e]
            x = block.norm1(hidden_states[:, s:e]) * (1.0 + scale_msa.index_select(0, idx)) + shift_msa.index_select(0, idx)
            q[:, s:e] = _apply_rotary_emb(attn.norm_q(attn.to_q(x).unflatten(-1, (attn.heads, -1))), cos[s:e], sin[s:e])
            k[:, s:e] = _apply_rotary_emb(attn.norm_k(attn.to_k(x).unflatten(-1, (attn.heads, -1))), cos[s:e], sin[s:e])
            v[:, s:e] = attn.to_v(x).unflatten(-1, (attn.heads, -1))
        # every query row attends to the full k / v independently, so the attention runs per chunk of queries and
        # its output is projected and added to the residual stream right away (q, k, v already hold every row)
        for s, e in spans:
            idx = adaln_indices[s:e]
            out = dispatch_attention_fn(q[:, s:e], k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                                        backend=attn.processor._attention_backend,
                                        parallel_config=attn.processor._parallel_config)
            hidden_states[:, s:e] = hidden_states[:, s:e] + gate_msa.index_select(0, idx) * attn.to_out[0](out.flatten(2, 3))
        del q, k, v, out
        for s, e in spans:
            idx = adaln_indices[s:e]
            h = hidden_states[:, s:e]
            x = block.norm2(h) * (1.0 + scale_mlp.index_select(0, idx)) + shift_mlp.index_select(0, idx)
            hidden_states[:, s:e] = h + gate_mlp.index_select(0, idx) * block.ff(x)
        return hidden_states

    return forward


class LowMemTransformer:
    """The H3 transformer with the DMAD LoRA: blocks streamed, AdaLN tabulated, the rest resident.

    `timesteps` is the list of `timestep` tensors the sampler will pass, one per step (the distinct noise levels of
    the packed sequence), so the tables can be built before sampling. `self.model` is called like the resident model.
    """

    def __init__(self, model_dir, lora_pairs, rank, alpha, timesteps, device, dtype=torch.bfloat16, cache_in_ram=False,
                 codec=None, cache_dir=None, log=None, chunk_rows=None):
        """`codec` ("int8") stores the streamed weights 8-bit: quantized on the way into host memory with
        `cache_in_ram`, else written once to `cache_dir` (default `~/.cache/dmad_h3/<checkpoint id>/<codec>`) and
        streamed from there. `chunk_rows` runs the transformer blocks row-chunked (`_chunked_block_forward`)."""
        import hashlib

        from accelerate import init_empty_weights
        from diffusers import MiniMaxH3Transformer3DModel
        from peft import LoraConfig, inject_adapter_in_model

        from .model import resolve_transformer_dir

        self.device = device
        tdir = resolve_transformer_dir(model_dir)
        with init_empty_weights(include_buffers=False):
            model = MiniMaxH3Transformer3DModel.from_config(MiniMaxH3Transformer3DModel.load_config(str(tdir)))
        keep_fp32 = model._keep_in_fp32_modules or []
        for name, p in list(model.named_parameters()):  # from_pretrained's mixed-precision rule
            want = torch.float32 if any(m in name.split(".") for m in keep_fp32) else dtype
            if p.dtype != want:
                module, attr = _find_param(model, name)
                module._parameters[attr] = nn.Parameter(torch.empty(p.shape, dtype=want, device="meta"), requires_grad=False)
        model.eval().requires_grad_(False)
        _buffers_to(model, device)

        config = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=sorted(lora_pairs))
        try:
            model = inject_adapter_in_model(config, model, adapter_name="default")
        except TypeError:
            model = inject_adapter_in_model(config, model)
        for module_name, pair in lora_pairs.items():
            module = model.get_submodule(module_name)
            for attr, value in (("lora_A", pair["A"]), ("lora_B", pair["B"])):
                linear = getattr(module, attr)["default"]
                _bind(linear, "weight", value.to(device=device, dtype=linear.weight.dtype))

        self.source = ShardIndex(tdir)
        blocks = {i: [] for i in range(len(model.transformer_blocks))}
        resident = []
        for key in self.source.keys():
            if key.startswith("transformer_blocks."):
                blocks[int(key.split(".")[1])].append(key)
            else:
                resident.append(key)
        load_resident(model, self.source, resident, device)
        self.model = model

        self._tabulate_adaln(timesteps, blocks)
        if chunk_rows:
            for block in model.transformer_blocks:
                block.forward = _chunked_block_forward(block, chunk_rows)
        streamed =[[k for k in blocks[i] if ".adaln_proj." not in k] for i in range(len(blocks))]
        if codec is not None and not cache_in_ram and cache_dir is None:
            checkpoint_id = hashlib.sha1(str(tdir).encode()).hexdigest()[:12]
            cache_dir = Path.home() / ".cache" / "dmad_h3" / checkpoint_id / codec
        self.streamer = Streamer(model, self.source, streamed, list(model.transformer_blocks), device, cache_in_ram,
                                 codec, cache_dir, log)

    @torch.no_grad()
    def _tabulate_adaln(self, timesteps, blocks):
        """adaln_proj(temb) for every block and step, computed exactly as the forward does, then the projection
        (260M parameters per block) is replaced by the table."""
        model = self.model
        temb_dtype = next(model.time_embedder.parameters()).dtype
        tembs = [model.time_embedder(model.time_proj(ts.to(self.device)).to(temb_dtype)) for ts in timesteps]
        for i, block in enumerate(model.transformer_blocks):
            proj = block.adaln_proj
            keys = [k for k in blocks[i] if ".adaln_proj." in k]
            load_resident(model, self.source, keys, self.device)
            tables = [torch.stack(proj(temb)) for temb in tembs]
            block.adaln_proj = AdaLNTables(tembs, tables)  # the projection's weights are dropped with `proj`
        del proj

    def close(self):
        self.streamer.close()
        self.model = self.source = None
        torch.cuda.empty_cache()
