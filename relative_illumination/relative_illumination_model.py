from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.base_model import ModelConfig
from torch import nn
from torch.nn import Parameter

from rawnerfacto import raw_utils
from rawnerfacto.rawnerfacto_model import RawNerfactoModel, RawNerfactoModelConfig
from rawnerfacto.utils import acc_loss, get_transmittance
from relative_illumination.relative_illumination_field import RelativeIlluminationField


@dataclass
class RelativeIlluminationModelConfig(RawNerfactoModelConfig):
    _target: type = field(default_factory=lambda: RelativeIlluminationModel)

    num_alpha_channels: Optional[Literal[1, 3]] = None
    """Number of alpha channels for illumination prediction. If 'None' will be set based on 'model_medium' (True=3/False/1). If set will overwride explicitly."""
    alpha_shift: float = 5.0
    """Alpha shifting (this allows the alpha output to be initialized at a controlled point)"""
    alpha_scale: float = 1.0
    """Alpha scaling (this allows the alpha to have broader, but limited range of values)."""
    use_surface_normal: bool = True
    """Whether to use surface normal also for alpha prediction."""
    model_medium: bool = True
    """Whether to model a medium between the camera and object."""
    background_color: Literal["random", "last_sample", "black", "white"] = "random"
    """Overwide the default background_color setting of Nerfacto to random."""
    average_init_medium_density: float = 1
    """Average init medium density."""
    use_transmittance_loss: bool = False
    """Whether to use transmittance loss."""
    transmittance_loss_mult: float = 0.0001
    """Transmittance loss multiplier."""
    big_field: bool = False
    """Use a larger illumiation field"""


class RelativeIlluminationModel(RawNerfactoModel):
    """Nerfacto model trained on raw images."""

    config: RelativeIlluminationModelConfig

    def __init__(self, config: ModelConfig, scene_box: SceneBox, num_train_data: int, **kwargs) -> None:
        config.background_color = "random" if config.model_medium else "black"
        super().__init__(
            config,
            scene_box,
            num_train_data,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # Fields
        self.illumination_field = RelativeIlluminationField(
            self.scene_box.aabb,
            num_alpha_channels=self.config.num_alpha_channels
            if self.config.num_alpha_channels is not None
            else 3
            if self.config.model_medium
            else 1,
            alpha_shift=self.config.alpha_shift,
            alpha_scale=self.config.alpha_scale,
            use_surface_normal=self.config.use_surface_normal,
            spatial_distortion=self.field.spatial_distortion,
            implementation=self.config.implementation,
            big=self.config.big_field,
        )

        # Medium parameters if medium is modeled.
        if self.config.model_medium:
            self.color_activation = lambda x: trunc_exp(x - 5)
            self.sigma_activation = nn.Softplus()
            self.medium_attn = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.medium_rgb = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.medium_bs = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["illumination_field"] = list(self.illumination_field.parameters())
        if self.config.model_medium:
            param_groups["fields"] += [
                self.medium_attn,
                self.medium_rgb,
                self.medium_bs,
            ]
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        outputs, field_outputs, ray_samples, weights = super().get_outputs_and_samples(ray_bundle)

        if self.training and self.camera_optimizer.config.mode != "off":
            # correct the camera poses applied to the illumination field
            # application is unusual but equivalent to apply_to_raybundle() (also see apply_to_camera())
            correction_matrices = self.camera_optimizer(ray_bundle.camera_indices.squeeze())
            adj = correction_matrices.expand(ray_samples.shape[-1], -1, -1, -1).moveaxis(0, 1).reshape(-1, 3, 4)
            c2w = ray_samples.metadata["c2ws"].reshape(-1, 3, 4)
            ray_samples.metadata["c2ws"] = torch.cat(
                [
                    torch.bmm(adj[..., :3, :3], c2w[..., :3, :3]),
                    c2w[..., :3, 3:] + adj[..., :3, 3:],
                ],
                dim=-1,
            ).reshape(ray_samples.shape + (3, 4))

        normals = (
            self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights).detach()
            if self.config.use_surface_normal and not self.config.predict_normals
            else None
        )

        illumination_field = self.illumination_field.forward(ray_samples, normals)["illumination_field"]

        if illumination_field.shape[-1] == 1:
            # if only one channel alpha
            illumination_field = illumination_field.expand_as(field_outputs[FieldHeadNames.RGB])

        if self.config.model_medium:
            medium_rgb = self.color_activation(self.medium_rgb).expand(*ray_samples.shape, 3)
            medium_bs = self.config.average_init_medium_density * self.sigma_activation(self.medium_bs).expand(
                *ray_samples.shape, 3
            )
            medium_attn = self.config.average_init_medium_density * self.sigma_activation(self.medium_attn).expand(
                *ray_samples.shape, 3
            )
            transmittance_obj = get_transmittance(ray_samples.deltas, field_outputs[FieldHeadNames.DENSITY])
            deltas_detached = ray_samples.deltas.detach()
            transmittance_attn = get_transmittance(deltas_detached, medium_attn)
            transmittance_bs = get_transmittance(deltas_detached, medium_bs)

            weights_obj = transmittance_attn * weights
            weights_obj = torch.nan_to_num(weights_obj)
            weights_medium = transmittance_obj * transmittance_bs * (1 - torch.exp(-medium_bs * deltas_detached))
            weights_medium = torch.nan_to_num(weights_medium)

            rgb_obj = self.renderer_rgb(
                rgb=illumination_field * field_outputs[FieldHeadNames.RGB],
                weights=weights_obj,
            )
            rgb_medium = self.renderer_rgb(rgb=illumination_field * medium_rgb, weights=weights_medium)

            rgb_train = rgb_obj + rgb_medium
        else:
            rgb_train = self.renderer_rgb(
                rgb=illumination_field * field_outputs[FieldHeadNames.RGB],
                weights=weights,
            )

        with torch.no_grad():
            alpha_map = self.renderer_rgb(rgb=illumination_field, weights=weights)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        outputs.update(
            {
                "rgb": rgb_train,
                "rgb_eval": rgb_train.clone(),
                "rgb_clean": rgb,
                "alpha_map": alpha_map,
            }
        )

        if self.config.model_medium:
            outputs["rgb_medium"] = rgb_medium
            outputs["rgb_clean_medium"] = self.renderer_rgb(rgb=medium_rgb, weights=weights_medium) + self.renderer_rgb(
                rgb=field_outputs[FieldHeadNames.RGB], weights=weights_obj
            )

        if self.config.model_medium and self.config.use_transmittance_loss:
            outputs["transmittance_obj"] = transmittance_obj

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training and self.config.model_medium and self.config.use_transmittance_loss:
            loss_dict["transmittance_loss"] = self.config.transmittance_loss_mult * acc_loss(
                outputs["transmittance_obj"], 100
            )

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        outputs = super().get_outputs_for_camera(camera, obb_box)

        exposure = (
            self.viewer_exposure_scale.value
            if self.viewer_exposure_scale.gui_handle is not None
            else self.config.exposure_value
            if self.config.exposure_value is not None
            else self.dataset_default_exposure
        )

        apply_srgb = (
            self.viewer_gamma_checkbox.value
            if self.viewer_gamma_checkbox.gui_handle is not None
            else self.config.apply_srgb
        )

        apply_color_transform = (
            self.viewer_color_transform_checkbox.value
            if self.viewer_color_transform_checkbox.gui_handle is not None
            else self.config.apply_color_transform
        )

        cam2rgb = self.cam2rgb if apply_color_transform else torch.eye(3, dtype=torch.float32, device=self.device)

        outputs["rgb_clean"] = raw_utils.postprocess_raw(outputs["rgb_clean"], cam2rgb, exposure, apply_srgb)
        outputs["alpha_map"] = raw_utils.postprocess_raw(
            outputs["alpha_map"], torch.eye(3).to(self.default_cam2rgb), exposure, False
        )
        if self.config.model_medium:
            outputs["rgb_medium"] = raw_utils.postprocess_raw(outputs["rgb_medium"], cam2rgb, exposure, apply_srgb)
            outputs["rgb_clean_medium"] = raw_utils.postprocess_raw(
                outputs["rgb_clean_medium"], cam2rgb, exposure, apply_srgb
            )

        # reorder outputs for viewer ux
        outputs_reordered = {
            "rgb": outputs["rgb"],
            "rgb_clean": outputs["rgb_clean"],
            "alpha_map": outputs["alpha_map"],
        }

        if self.config.model_medium:
            outputs_reordered["rgb_medium"] = outputs["rgb_medium"]
            outputs_reordered["rgb_clean_medium"] = outputs["rgb_clean_medium"]

        for key, value in outputs.items():
            if key not in outputs_reordered:
                outputs_reordered[key] = value

        return outputs_reordered
