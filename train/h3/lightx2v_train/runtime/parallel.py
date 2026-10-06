from lightx2v_train.runtime.distributed import is_distributed
from lightx2v_train.runtime.fsdp import apply_fsdp2, fsdp2_enabled


def apply_parallel(model, config):
    """Shard the model with FSDP2 when running distributed."""
    if not is_distributed():
        return model
    if not fsdp2_enabled(config):
        raise RuntimeError("distributed training requires distributed.fsdp2.enabled: true")
    return apply_fsdp2(model, config)


def set_parallel_gradient_sync(model, enabled):
    model.set_fsdp2_gradient_sync(enabled)
