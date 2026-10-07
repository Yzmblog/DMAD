"""ShardIndex reads safetensors shards with pread on Windows and with the default mmap elsewhere.

Run: python -m pytest tests/test_lowmem_shard_index.py  (CPU only; tiny synthetic shards, no model weights)
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import save_file

LOWMEM = Path(__file__).resolve().parents[1] / "dmad_h3" / "lowmem.py"
PLATFORMS = [("win32", {"backend": "pread"}), ("linux", {}), ("darwin", {})]


@pytest.fixture(scope="module")
def lowmem():
    # Load the file itself: `import dmad_h3` would import the H3 model stack (diffusers, transformers).
    spec = importlib.util.spec_from_file_location("dmad_h3_lowmem", LOWMEM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def safe_open_calls(lowmem, monkeypatch):
    calls = []
    real = lowmem.safe_open

    def recording(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(lowmem, "safe_open", recording)
    return calls


def write_shards(directory, with_index):
    shards = {
        "model-00001-of-00002.safetensors": {
            "a.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "a.bias": torch.arange(4, dtype=torch.bfloat16),
        },
        "model-00002-of-00002.safetensors": {"b.weight": torch.arange(6, dtype=torch.float16).reshape(2, 3)},
    }
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(directory / name))
        weight_map.update(dict.fromkeys(tensors, name))
    if with_index:
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return {key: value for tensors in shards.values() for key, value in tensors.items()}


@pytest.mark.parametrize("with_index", [True, False], ids=["index", "no-index"])
@pytest.mark.parametrize(("platform", "expected_kwargs"), PLATFORMS, ids=[platform for platform, _ in PLATFORMS])
def test_shard_index_backend_and_contents(lowmem, safe_open_calls, monkeypatch, tmp_path, platform, expected_kwargs, with_index):
    monkeypatch.setattr(sys, "platform", platform)
    written = write_shards(tmp_path, with_index)

    index = lowmem.ShardIndex(tmp_path)

    assert sorted(index.keys()) == sorted(written)
    for key, value in written.items():
        assert index.info(key) == (tuple(value.shape), value.dtype)
        assert torch.equal(index.tensor(key), value)
        assert torch.equal(index.tensor(key), value)  # the retained handle is read again
    assert len(index._handles) == 2  # one retained handle per shard
    assert len(safe_open_calls) == (2 if with_index else 4)  # the key scan opens each shard once more
    assert all(kwargs == expected_kwargs for kwargs in safe_open_calls)


@pytest.mark.parametrize("platform", [platform for platform, _ in PLATFORMS])
def test_damaged_shard_is_rejected(lowmem, monkeypatch, tmp_path, platform):
    monkeypatch.setattr(sys, "platform", platform)
    write_shards(tmp_path, with_index=True)
    shard = tmp_path / "model-00002-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes()[:-4])  # truncated tensor data

    index = lowmem.ShardIndex(tmp_path)

    with pytest.raises(SafetensorError):
        index.tensor("b.weight")
