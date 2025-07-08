from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.viewer.viewer_elements import ViewerCheckbox, ViewerSlider

from rawnerfacto import raw_utils
from rawnerfacto.rawnerfacto_field import RawNerfactoField


@dataclass
class RawNerfactoModelConfig(NerfactoModelConfig):
    _target: type = field(default_factory=lambda: RawNerfactoModel)

    learn_exposure_scale: bool = False
    """Whether to learn the exposure scaling factor during training."""
    default_white_balance: Tuple[float, ...] = (0.453097, 1, 0.746356)
    """Default white balance coefficient for raw image post-processing."""
    default_color_matrix2: Tuple[float, ...] = (
        1.0344,
        -0.421,
        -0.062,
        -0.2315,
        1.0625,
        0.1948,
        0.0093,
        0.1058,
        0.5541,
    )
    """Default ColorMatrix2 for raw image post-processing."""
    use_appearance_embedding: bool = False
    """RawNeRF should disable appearance embedding always."""
    background_color: Literal["random", "last_sample", "black", "white"] = "random"
    """Overwide the default background_color setting of Nerfacto to random."""
    enable_view_dependence: bool = True
    """Whether to use viewing directions as input for object colors."""
    exposure_value: Optional[float] = None
    """Set an explicit exposure value. Is overridden in the viewer by the exposure slider. Mostly relevant for rendering."""
    apply_srgb: bool = False
    """Whether to apply srgb to final output. Is overriden in the viewer by the checkbox. Mostly relevant for rendering."""
    apply_color_transform: bool = False
    """Whether to apply color transformation to final output. Is overriden in the viewer by the checkbox. Mostly relevant for rendering."""


class RawNerfactoModel(NerfactoModel):
    """Nerfacto model trained on raw images."""

    config: RawNerfactoModelConfig

    def __init__(self, config: ModelConfig, scene_box: SceneBox, num_train_data: int, **kwargs) -> None:
        super().__init__(config, scene_box, num_train_data, **kwargs)
        self.viewer_exposure_scale = ViewerSlider(
            name="Exposure", default_value=0.5, min_value=0.0, max_value=3, step=0.01
        )
        self.viewer_gamma_checkbox = ViewerCheckbox(name="Apply sRGB Gamma", default_value=False)
        self.viewer_color_transform_checkbox = ViewerCheckbox(name="Apply Color Transform", default_value=False)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # Fields
        self.field = RawNerfactoField(
            self.scene_box.aabb,
            learned_exposure_scaling=self.config.learn_exposure_scale,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            enable_view_dependence=self.config.enable_view_dependence,
            implementation=self.config.implementation,
        )

        default_wb = torch.tensor(self.config.default_white_balance)
        cam2camwb = torch.diag(1.0 / default_wb)
        # ColorMatrix2 converts from XYZ color space to "reference illuminant" (white
        # balanced) camera space.
        xyz2camwb = torch.tensor(self.config.default_color_matrix2).reshape(3, 3)
        rgb2cam2b = xyz2camwb @ torch.from_numpy(raw_utils._RGB2XYZ).to(torch.float32)
        # We normalize the rows of the full color correction matrix, as is done in
        # https://github.com/AbdoKamel/simple-camera-pipeline.
        rgb2cam2b = rgb2cam2b / rgb2cam2b.sum(dim=-1, keepdim=True)
        self.register_buffer("default_cam2rgb", torch.linalg.inv(rgb2cam2b) @ cam2camwb)

        self.cam2rgb = None
        self.dataset_default_exposure = None

        self.bayer_mask = (
            self.kwargs["metadata"]["bayer_mask"].cuda() if "bayer_mask" in self.kwargs["metadata"] else None
        )

    # this method mainly exists to allow the relative-illumination field to share the same code without having to relalculate raysamples etc.
    def get_outputs_and_samples(self, ray_bundle: RayBundle):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        use_surface_normal = (
            self.config.predict_normals
            if not hasattr(self.config, "use_surface_normal")
            else (self.config.use_surface_normal or self.config.predict_normals)
        )
        field_outputs = self.field.forward(
            ray_samples,
            compute_normals=use_surface_normal,
        )
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        outputs = {}

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs.update(
            {
                "rgb": rgb,
                "rgb_eval": rgb.clone(),
                "accumulation": accumulation,
                "depth": depth,
                "expected_depth": expected_depth,
            }
        )

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS],
                ray_bundle.directions,
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        with torch.no_grad():
            if self.cam2rgb is None:
                self.cam2rgb = (
                    ray_bundle.metadata["cam2rgb"][0].reshape(3, 3)
                    if "cam2rgb" in ray_bundle.metadata
                    else self.default_cam2rgb
                )

            if self.dataset_default_exposure is None and "exposure" in ray_bundle.metadata:
                self.dataset_default_exposure = ray_bundle.metadata["exposure"][0]
        return outputs, field_outputs, ray_samples, weights

    def get_outputs(self, ray_bundle: RayBundle):
        return self.get_outputs_and_samples(ray_bundle)[0]

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb_eval"]

        if self.bayer_mask is not None:
            # TODO: can't demosiac gt_rgb because the image is incomplete
            # gt_rgb = raw_utils.bilinear_demosaic(gt_rgb[..., 0].cpu().numpy())
            # gt_rgb = torch.from_numpy(gt_rgb).to(predicted_rgb)
            metrics_dict["psnr"] = 0
        else:
            metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        # RawNeRF weighted L2 loss and bayer mask
        _, y, x = torch.split(batch["indices"], 1, dim=-1)  # Must be single, not downscaled camera
        pred_rgb_clip = torch.minimum(torch.ones_like(pred_rgb), pred_rgb)
        gt_rgb = torch.minimum(torch.ones_like(gt_rgb), gt_rgb)
        scaling_grad = 1.0 / (1e-3 + pred_rgb_clip.detach())
        gt_rgb = scaling_grad * gt_rgb
        pred_rgb_clip = scaling_grad * pred_rgb_clip
        if self.bayer_mask is not None:
            bayer_mask = self.bayer_mask[y, x].squeeze(1)
            gt_rgb = bayer_mask * gt_rgb
            pred_rgb_clip = bayer_mask * pred_rgb_clip

        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb_clip)
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)
        ray_bundle.metadata["c2ws"] = camera.camera_to_worlds.reshape(-1, 12)
        outputs = self.get_outputs_for_camera_ray_bundle(ray_bundle)

        # Additionally post process raw rgb image into linear sRGB image
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
        outputs["rgb"] = raw_utils.postprocess_raw(outputs["rgb"], cam2rgb, exposure, apply_srgb)

        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb_eval"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        if self.bayer_mask is not None:
            # demosiac gt_rgb for evaluation metrics in the case of raw data
            gt_rgb = raw_utils.bilinear_demosaic(gt_rgb[..., 0].cpu().numpy())
            gt_rgb = torch.from_numpy(gt_rgb).to(predicted_rgb)

        # TODO: Post-process the image for visualization?
        gt_rgb = torch.minimum(torch.ones_like(gt_rgb), gt_rgb)
        predicted_rgb = torch.minimum(torch.ones_like(predicted_rgb), predicted_rgb)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
