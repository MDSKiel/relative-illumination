from typing import Dict, Literal, Optional

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.fields.nerfacto_field import NerfactoField
from torch import Tensor, nn


class RawNerfactoField(NerfactoField):
    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        learned_exposure_scaling: bool = True,
        geo_feat_dim: int = 16,
        appearance_embedding_dim: int = 32,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        enable_view_dependence: bool = False,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        **kwargs,
    ) -> None:
        super().__init__(
            aabb,
            num_images,
            geo_feat_dim=geo_feat_dim,
            appearance_embedding_dim=appearance_embedding_dim,
            num_layers_color=num_layers_color,
            hidden_dim_color=hidden_dim_color,
            **kwargs,
        )
        self.learned_exposure_scaling = learned_exposure_scaling
        self.enable_view_dependence = enable_view_dependence

        # The activation function of raw field is biased exponential
        dir_enc_out_dim = self.direction_encoding.get_out_dim() if self.enable_view_dependence else 0
        self.mlp_head = MLP(
            in_dim=self.geo_feat_dim + self.appearance_embedding_dim + dir_enc_out_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        # learned exposure scaling
        if self.learned_exposure_scaling:
            max_num_exposures = 1000
            self.exposure_scaling_offsets = nn.Embedding(max_num_exposures, 3, dtype=torch.float32)
            self.exposure_scaling_offsets.weight.data.fill_(0)

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        if self.enable_view_dependence:
            h = torch.cat(
                [d, density_embedding.view(-1, self.geo_feat_dim)]
                + (
                    [embedded_appearance.view(-1, self.appearance_embedding_dim)]
                    if embedded_appearance is not None
                    else []
                ),
                dim=-1,
            )
        else:
            h = torch.cat(
                [density_embedding.view(-1, self.geo_feat_dim)]
                + (
                    [embedded_appearance.view(-1, self.appearance_embedding_dim)]
                    if embedded_appearance is not None
                    else []
                ),
                dim=-1,
            )

        rgb_before_activation = self.mlp_head(h).view(*outputs_shape, -1).to(directions)

        # Biased exponential activation function: Make sure that the RGB output can be larger than 1
        rgb = trunc_exp(rgb_before_activation - 5.0)
        # Exposure scaling logic copied from RawNeRF:
        has_exposure_info = "exposure_idx" in ray_samples.metadata
        if has_exposure_info and self.training and self.learned_exposure_scaling:
            # Force scaling offset to always be zero when exposure_idx is 0.
            # This constraint fixes a reference point for the scene's brightness.
            exposure_idx = ray_samples.metadata["exposure_idx"][..., 0, :].squeeze(-1)
            mask = exposure_idx > 0
            # Scaling is parameterized as an offset from 1.
            scaling = 1 + mask[..., None] * self.exposure_scaling_offsets(exposure_idx)
            rgb = rgb * scaling[..., None, :]

        outputs.update({FieldHeadNames.RGB: rgb})
        return outputs
