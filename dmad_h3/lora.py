"""DMAD student LoRAs: reading the safetensors file and attaching it to the transformer with PEFT.

The released files use the Diffusers LoRA layout, `<module>.lora.down.weight` (A, [rank, in]) and
`<module>.lora.up.weight` (B, [out, rank]), with `<module>` the transformer's own module name, e.g.
`transformer_blocks.3.attn.to_q`; the adapter adds `(alpha / rank) * B A x`, and the released LoRAs have `alpha == rank`.
PEFT-style keys (`lora_A` / `lora_B`, optional `transformer.` prefix) are accepted too.
"""

import re

from safetensors import safe_open

_SUFFIXES = {
    ".lora.down.weight": "A",
    ".lora.up.weight": "B",
    ".lora_A.weight": "A",
    ".lora_B.weight": "B",
    ".lora_A.default.weight": "A",
    ".lora_B.default.weight": "B",
}


def read_lora_file(path):
    """-> (pairs, metadata): pairs maps module name -> {"A": tensor, "B": tensor}."""
    pairs, meta = {}, {}
    with safe_open(str(path), "pt") as f:
        meta = dict(f.metadata() or {})
        for key in f.keys():
            for suffix, part in _SUFFIXES.items():
                if key.endswith(suffix):
                    module = key[: -len(suffix)]
                    module = module[len("transformer.") :] if module.startswith("transformer.") else module
                    pairs.setdefault(module, {})[part] = f.get_tensor(key)
                    break
            else:
                raise ValueError(f"Unexpected tensor in the LoRA file: {key}")
    if not pairs:
        raise ValueError(f"No LoRA pairs found in {path}")
    incomplete = [m for m, p in pairs.items() if set(p) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"LoRA modules missing A or B: {incomplete[:5]}")
    return pairs, meta


def lora_rank_alpha(pairs, meta, alpha=None):
    ranks = {p["A"].shape[0] for p in pairs.values()}
    if len(ranks) != 1:
        raise ValueError(f"Mixed LoRA ranks: {sorted(ranks)}")
    rank = ranks.pop()
    if alpha is None:
        alpha = float(meta["lora_alpha"]) if "lora_alpha" in meta else float(rank)
    return rank, float(alpha)


def attach_lora(transformer, pairs, rank, alpha):
    """Inject a PEFT LoRA adapter for exactly the modules in `pairs` and load the weights (no fusion)."""
    from peft import LoraConfig, inject_adapter_in_model
    from peft.utils import set_peft_model_state_dict

    config = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=sorted(pairs))
    try:
        transformer = inject_adapter_in_model(config, transformer, adapter_name="default")
    except TypeError:
        transformer = inject_adapter_in_model(config, transformer)
    state = {}
    for module, p in pairs.items():
        state[f"{module}.lora_A.weight"] = p["A"]
        state[f"{module}.lora_B.weight"] = p["B"]
    result = set_peft_model_state_dict(transformer, state)
    if result and result.unexpected_keys:
        raise RuntimeError(f"unexpected LoRA keys: {result.unexpected_keys[:5]}")
    loaded = {n for n, _ in transformer.named_parameters() if re.search(r"\.lora_[AB]\.", n)}
    if len(loaded) != 2 * len(pairs):
        raise RuntimeError(f"expected {2 * len(pairs)} LoRA parameters after injection, found {len(loaded)}")
    return transformer

