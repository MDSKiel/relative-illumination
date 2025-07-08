from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from rawnerfacto.colmap_raw_image_dataparser import ColmapRawImageDataParserConfig
from rawnerfacto.raw_image_dataset import RawImageDataset
from relative_illumination.relative_illumination_model import RelativeIlluminationModelConfig

relative_illumination = MethodSpecification(
    TrainerConfig(
        method_name="relative-illumination",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        log_gradients=False,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                _target=ParallelDataManager[RawImageDataset],
                dataparser=ColmapRawImageDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=RelativeIlluminationModelConfig(
                eval_num_rays_per_chunk=1 << 13,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                big_field=False
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
            "illumination_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Config for relative-illumination",
)

relative_illumination_big = MethodSpecification(
    TrainerConfig(
        method_name="relative-illumination-big",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        log_gradients=False,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                _target=ParallelDataManager[RawImageDataset],
                dataparser=ColmapRawImageDataParserConfig(),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=4096,
            ),
            model=RelativeIlluminationModelConfig(
                eval_num_rays_per_chunk=1 << 13,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
                num_proposal_samples_per_ray=(512, 256),
                num_nerf_samples_per_ray=128,
                hidden_dim=128,
                hidden_dim_color=128,
                max_res=4096,
                log2_hashmap_size=21,
                big_field=True
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
            "illumination_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Config for relative-illumination",
)
