"""Webdataset loader for DMAD: the shards of rcm/datasets/webdataset.py plus caption-paired real latents
(`real_latent.pt`, see rcm/datasets/build_real_latents.py), surfaced as data_batch["real_latents"]."""

import glob
from itertools import islice

import webdataset as wds
from torch.utils.data import DataLoader
from webdataset import utils as wds_utils

from rcm.datasets.webdataset import dict_collation_fn


def _split_by_dp_group(cp_size):
    """Split shards by data-parallel rank (rank // cp_size): all ranks of a context-parallel group read the same shards
    (the group consumes one sample, broadcast from its first rank), and the groups partition the shards. At cp_size=1
    this is wds.split_by_node."""

    def split(src):
        rank, world_size, _worker, _num_workers = wds_utils.pytorch_worker_info()
        if world_size % cp_size != 0:
            raise ValueError(f"world_size {world_size} not divisible by cp_size {cp_size}")
        dp_rank, dp_world = rank // cp_size, world_size // cp_size
        if dp_world > 1:
            yield from islice(src, dp_rank, None, dp_world)
        else:
            yield from src

    return split


def create_dataloader_dmad(tar_path_pattern, batch_size, num_workers=8, shuffle_buffer=1000, prefetch_factor=2):
    """Shards are split across data-parallel ranks and dataloader workers before the shard shuffle, so every shard is
    read exactly once per pass."""
    from megatron.core import parallel_state

    cp_size = parallel_state.get_context_parallel_world_size() if parallel_state.is_initialized() else 1
    shards = glob.glob(tar_path_pattern)
    if not shards:
        raise FileNotFoundError(f"No files found with pattern '{tar_path_pattern}'")

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shards),
        _split_by_dp_group(cp_size),
        wds.split_by_worker,
        wds.shuffle(1000),
        wds.tarfile_to_samples(),
        wds.shuffle(shuffle_buffer),
        wds.decode(wds.handle_extension("pt", wds.torch_loads)),
        wds.rename(
            latents="latent.pt",
            t5_text_embeddings="embed.pt",
            prompts="prompt.txt",
            real_latents="real_latent.pt",
        ),
        wds.batched(batch_size, partial=False, collation_fn=dict_collation_fn),
    )
    return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
