"""DMAD experiments on Wan2.1 T2V 480p: rCM's recipe with the DMD term replaced by the two-head GAN critic
(rcm/models/t2v_model_distill_dmad.py) and no sCM loss.

    wan2pt1_1pt3B_res480p_t2v_dmad : 1.3B, 8 GPUs, batch 8, FSDP shard 4
    wan2pt1_14B_res480p_t2v_dmad   : 14B, 64 GPUs, context parallel 8 (batch 8), FSDP shard 32

Data: webdataset shards whose rows hold the teacher-sample latent (latent.pt), the text embedding (embed.pt), the prompt
and a caption-paired real-video latent (real_latent.pt); see rcm/datasets/build_real_latents.py.
"""

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from rcm.callbacks.dmad_scalars import DMADScalars
from rcm.configs.experiments.rcm.wan2pt1_t2v import build_debug_run
from rcm.datasets.webdataset_dmad import create_dataloader_dmad
from rcm.models.t2v_model_distill_dmad import T2VDistillConfig_DMAD, T2VDistillModel_DMAD

WEBDATASET_LOADER_DMAD = L(create_dataloader_dmad)(
    tar_path_pattern="/path/to/shards/shard*.tar",
    batch_size=1,
    num_workers=8,
    shuffle_buffer=1000,
    prefetch_factor=2,
)

FSDP_CONFIG_T2V_DISTILL_DMAD = dict(
    trainer=dict(distributed_parallelism="fsdp"),
    model=L(T2VDistillModel_DMAD)(config=T2VDistillConfig_DMAD(fsdp_shard_size=8), _recursive_=False),
)

WAN2PT1_1PT3B_RES480P_T2V_DMAD: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/wan2pt1_1pt3B_res480p_t2v_rCM",
            {"override /model": "fsdp_t2v_distill_dmad"},
            {"override /data_train": "webdataset_dmad"},
            "_self_",
        ],
        job=dict(group="dmad_Wan", name="wan2pt1_1pt3B_res480p_t2v_dmad"),
        optimizer=dict(betas=(0.0, 0.99)),
        model=dict(
            config=dict(
                state_t=20,  # 77 frames
                loss_scale=0.0,  # no sCM loss
                tangent_warmup=0,
                # generator and critic updates alternate 1:1 (the generator steps when (iter - warmup) % freq == 0)
                student_update_freq=2,
                optimizer_fake_score=dict(lr=2e-6, betas=(0.0, 0.99)),
                fsdp_shard_size=4,
                # the critic step runs three fake-score forwards (G, Q, T): checkpoint every block
                net_fake_score=dict(sac_config=dict(mode="block_wise")),
                dmad_critic_spectral_norm=True,
                dmad_generator_spectral_norm=True,
                dmad_sn_generator_frac=0.5,
                dmad_teacher_gap_route=True,
                dmad_gap_tau=2.0,
            )
        ),
        trainer=dict(
            callbacks=dict(
                dmad_scalars=L(DMADScalars)(every_n=50, barrier_after_run=False),
            )
        ),
    ),
    flags={"allow_objects": True},
)

WAN2PT1_14B_RES480P_T2V_DMAD: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/wan2pt1_1pt3B_res480p_t2v_dmad",
            {"override /net": "wan2pt1_14B_t2v_jvp"},
            {"override /net_teacher": "wan2pt1_14B_t2v"},
            {"override /net_fake_score": "wan2pt1_14B_t2v"},
            "_self_",
        ],
        job=dict(group="dmad_Wan", name="wan2pt1_14B_res480p_t2v_dmad"),
        optimizer=dict(lr=1e-6, weight_decay=0.01, betas=(0.0, 0.99)),
        model=dict(
            config=dict(
                fsdp_shard_size=32,
                dmad_sn_staged_build=True,
                teacher_ckpt="assets/checkpoints/Wan2.1-T2V-14B.dcp",
                optimizer_fake_score=dict(lr=1e-6, weight_decay=0.01, betas=(0.0, 0.99)),
                net=dict(sac_config=dict(mode="mm_only")),
            )
        ),
        checkpoint=dict(save_iter=500),
        trainer=dict(
            callbacks=dict(
                # no in-training sample videos at 14B
                every_n_sample_reg=dict(every_n=100000000, run_at_start=False),
                every_n_sample_ema=dict(every_n=100000000),
            )
        ),
        model_parallel=dict(context_parallel_size=8),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
cs.store(group="model", package="_global_", name="fsdp_t2v_distill_dmad", node=FSDP_CONFIG_T2V_DISTILL_DMAD)
cs.store(group="data_train", package="dataloader_train", name="webdataset_dmad", node=WEBDATASET_LOADER_DMAD)
for job in [WAN2PT1_1PT3B_RES480P_T2V_DMAD, WAN2PT1_14B_RES480P_T2V_DMAD]:
    cs.store(group="experiment", package="_global_", name=job["job"]["name"], node=job)
    cs.store(group="experiment", package="_global_", name=job["job"]["name"] + "_debug", node=build_debug_run(job))
