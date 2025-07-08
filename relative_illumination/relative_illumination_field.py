from typing import Dict, Literal, Optional

import nerfstudio.utils.poses as pose_utils
import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import get_normalized_directions
from torch import Tensor, nn


class RelativeIlluminationField(nn.Module):
    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_levels: int = 8,
        base_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 17,
        features_per_level: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 32,
        geo_feat_dim: int = 16,
        num_layers_head: int = 2,
        hidden_dim_head: int = 64,
        num_alpha_channels: Literal[1, 3] = 1,
        alpha_shift: float = 5.0,
        alpha_scale: float = 1.0,
        use_surface_normal: Optional[bool] = False,
        big: Optional[bool] = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        num_levels = 16 if big else num_levels
        max_res = 2048 if big else max_res
        hidden_dim = 64 if big else hidden_dim


        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        self.num_alpha_channels = num_alpha_channels
        self.use_surface_normal = use_surface_normal
        self.register_buffer("alpha_shift", torch.tensor(alpha_shift))
        self.register_buffer("alpha_scale", torch.tensor(alpha_scale))

        # MLP base
        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        if self.use_surface_normal:
            self.direction_encoding = SHEncoding(
                levels=4,
                implementation=implementation,
            )

        # MLP head
        h_dim = (
            self.geo_feat_dim + self.direction_encoding.get_out_dim() if self.use_surface_normal else self.geo_feat_dim
        )
        self.mlp_head = MLP(
            in_dim=h_dim,
            num_layers=num_layers_head,
            layer_width=hidden_dim_head,
            out_dim=num_alpha_channels,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

    def forward(
        self,
        ray_samples: RaySamples,
        normals: Optional[Float[Tensor, "*bs 3"]] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        # Get all sample positions.
        positions = ray_samples.frustums.get_positions()
        w2cs = pose_utils.inverse(ray_samples.metadata["c2ws"][:, 0, ...].view(-1, 3, 4)).unsqueeze(1)
        # Convert all sample positions to the local camera coordinate space
        positions = (torch.matmul(w2cs[..., :3, :3], positions.unsqueeze(-1)) + w2cs[..., :3, 3:]).squeeze(-1)

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)

        if self.use_surface_normal:
            # Convert all normals to the local camera coordinate space
            w2cs = w2cs.squeeze(1)
            n_cam = get_normalized_directions((torch.matmul(w2cs[..., :3, :3], normals.unsqueeze(-1))).squeeze(-1))
            d = self.direction_encoding(n_cam)
            d = d[..., None, :].expand(*ray_samples.frustums.shape, d.shape[-1]).reshape(-1, d.shape[-1])

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat)

        if self.use_surface_normal:
            h = torch.cat([h, d], dim=-1)

        mlp_head = self.mlp_head(h).view(*ray_samples.frustums.shape, -1).to(positions)
        illumination = self.alpha_scale * torch.sigmoid(mlp_head - self.alpha_shift)
        illumination = illumination * selector[..., None]

        outputs.update({"illumination_field": illumination})
        return outputs
