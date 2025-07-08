"""
Data parser for COLMAP with raw image data.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import Imath
import numpy as np
import OpenEXR as exr
import torch
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser, ColmapDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE

from rawnerfacto import raw_utils

try:
    import rawpy
except ImportError:
    import newrawpy as rawpy  # type: ignore


@dataclass
class ColmapRawImageDataParserConfig(ColmapDataParserConfig):
    _target: Type = field(default_factory=lambda: ColmapRawImageDataParser)
    raw_path: Path = Path("raw")
    """Path to raw images directory relative to the data path."""
    exposure_percentile: Optional[float] = 97.0
    """Image percentile to expose as white."""
    downscale_factor: Optional[int] = None
    """Parameter can not be used for training, only valid for rendering."""
    max_number_of_images: Optional[int] = None
    """If set will only take the first n images out of the raw directory."""
    disable_raw: bool = False
    """Act as the default colmap parser but add c2w poses to the camera metadata (relevant for illumination field)"""
    read_exr: bool = False
    """Read exr images"""
    # These can disable pose modifications, e.g. to retain reconstruction coordinate system
    # center_method: Literal["poses", "focus", "none"] = "none"
    # orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    # assume_colmap_world_coordinate_convention: bool = False
    # auto_scale_poses: bool = False


class ColmapRawImageDataParser(ColmapDataParser):
    """COLMAP Raw Image DatasetParser.
    Expects a folder with the following structure:
        images/ # folder containing sRGB images used to create the COLMAP model (This is generally not used)
        sparse/0 # folder containing the COLMAP reconstruction (either TEXT or BINARY format)
        raw/ # folder containing raw images (a pair of DNG image and a JSON file for metadata)
        masks/ # (OPTIONAL) folder containing masks for each image
        depths/ # (OPTIONAL) folder containing depth maps for each image
    """

    config: ColmapRawImageDataParserConfig

    def __init__(self, config: ColmapRawImageDataParserConfig):
        super().__init__(config)
        if not self.config.disable_raw and self.config.downscale_factor is not None:
            CONSOLE.print("[bold yellow]Warning: Use downscale_factor only during rendering!")

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        if self.config.disable_raw:
            outputs = super()._generate_dataparser_outputs(split)
            outputs.cameras.metadata = {"c2ws": outputs.cameras.camera_to_worlds.reshape(-1, 12)}
            return outputs

        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."
        raw_path = self.config.data / self.config.raw_path
        assert raw_path.exists(), f"Raw path {raw_path} does not exist."

        # Determine the raw image extension. Supported format is [".raw"]
        def get_raw_ext(raw_path, allowed_exts):
            # Hard set recursive as True
            recursive = True
            glob_str = "**/[!.]*" if recursive else "[!.]*"
            raw_paths = [p for p in raw_path.glob(glob_str) if p.suffix.lower() in allowed_exts]
            assert len(raw_paths) > 0, "Raw path does not contain any raw images."
            return raw_paths[0].suffix

        raw_ext = get_raw_ext(raw_path, [".dng"] if not self.config.read_exr else [".exr"])

        meta = self._get_all_images_and_cameras(colmap_path)
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []
        # Read the metadata of the raw image from ".json" files
        exif_filenames = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in (
            meta["frames"]
            if self.config.max_number_of_images is None
            else meta["frames"][: self.config.max_number_of_images]
        ):
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            # swap the image file name extension by the raw extension
            fname = Path(frame["file_path"]).with_suffix(raw_ext).name
            image_filenames.append(self.config.data / self.config.raw_path / fname)
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (len(mask_filenames) == len(image_filenames)), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (len(depth_filenames) == len(image_filenames)), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)

        # Exif files should be read before the split. Because it contains all meta information of the entire dataset.
        exif_filenames = [path.with_suffix(".json") for path in image_filenames]

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        metadata = {}
        cam_meta = {"c2ws": poses[:, :3, :4].reshape(-1, 12).to(dtype=torch.float32)}
        # only attempt to read exposure etc. from exif if raw (i.e. not processing exr)
        if self.config.read_exr:  # exr
            global_max = 0
            global_min = sys.maxsize
            for file in image_filenames:
                exrfile = exr.InputFile(file.as_posix())
                header = exrfile.header()
                if "white_level" in header and "black_level" in header:
                    maximum = float(header["white_level"])
                    minimum = float(header["black_level"])
                else:
                    R = np.frombuffer(exrfile.channel("R", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
                    G = np.frombuffer(exrfile.channel("G", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
                    B = np.frombuffer(exrfile.channel("B", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
                    img = np.stack([R, G, B])
                    maximum = float(np.quantile(img, 0.99))
                    minimum = float(np.min(img))
                exrfile.close()
                global_max = max(global_max, maximum)
                global_min = min(global_min, minimum)
            cam_meta.update({"white_level": global_max, "black_level": global_min})

        else:  # raw
            # Before constructing the Camera dataparser output. We need to process the exif data to obtain the meta information of the raw images.
            # The implementation is taken from: https://github.com/google-research/multinerf/blob/main/internal/raw_utils.py
            def read_exif(p):
                with open(p.as_posix(), "rb") as f:
                    json_data = json.load(f)  # data can be nested or toplevel
                    exif = json_data[0] if len(json_data) == 1 else json_data
                    return exif

            tmp_cam_meta = raw_utils.process_exif([read_exif(p) for p in exif_filenames])

            # Next we determine an index for each unique shutter speed in the data.
            shutter_speeds = tmp_cam_meta["ShutterSpeed"]
            # Sort the shutter speeds from slowest (largest) to fastest (smallest).
            # This way index 0 will always correspond to the brightest image.
            ## !! Note: in our experiment, we actually need to sort it from smallest to largest
            unique_shutters = np.sort(np.unique(shutter_speeds))
            exposure_idx = np.zeros_like(shutter_speeds, dtype=np.int32)
            for i, shutter in enumerate(unique_shutters):
                # Assign index `i` to all images with shutter speed `shutter`.
                exposure_idx[shutter_speeds == shutter] = i
            # Rescale to use relative shutter speeds, where 1. is the brightest.
            # This way the NeRF output with exposure=1 will always be reasonable.
            exposure_values = shutter_speeds / unique_shutters[0]

            # Rescale raw sensor measurements to [0, 1] (plus noise).
            black_level = (
                tmp_cam_meta["BlackLevel"]
                if tmp_cam_meta["BlackLevel"].ndim == 1
                else np.mean(tmp_cam_meta["BlackLevel"], axis=-1)
            )
            white_level = tmp_cam_meta["WhiteLevel"]
            cam2rgb = tmp_cam_meta["cam2rgb"]
            bayer_pattern = tmp_cam_meta["CFAPattern2"]
            # Calculate value for exposure level when gamma mapping, defaults to 97%.
            # Always based on full resolution image 0 (for consistency).
            if len(image_filenames) > 0:
                with open(image_filenames[0].as_posix(), "rb") as f:
                    raw0 = rawpy.imread(f).raw_image
                    raw0 = raw0.astype(np.float32)
                    im0 = (raw0 - black_level[0]) / (white_level[0] - black_level[0])
                    im0 = raw_utils.bilinear_demosaic(im0, bayer_pattern[0].tolist())
                    im0_rgb = np.matmul(im0, cam2rgb[0].T)
                    exposure = np.percentile(im0_rgb, self.config.exposure_percentile)
            else:
                exposure = 0.5

            cam_meta.update(
                {
                    "black_level": torch.from_numpy(black_level)[idx_tensor].unsqueeze(-1).to(dtype=torch.float32),
                    "white_level": torch.from_numpy(white_level)[idx_tensor].unsqueeze(-1).to(dtype=torch.float32),
                    "cam2rgb": torch.from_numpy(cam2rgb)[idx_tensor].reshape(-1, 9).to(dtype=torch.float32),
                    "exposure": torch.tensor([exposure], dtype=torch.float32),
                    "exposure_values": torch.from_numpy(exposure_values)[idx_tensor]
                    .unsqueeze(-1)
                    .to(dtype=torch.float32),
                    "exposure_idx": torch.from_numpy(exposure_idx)[idx_tensor].unsqueeze(-1),
                    "bayer_pattern": torch.from_numpy(bayer_pattern)[idx_tensor],
                }
            )

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=cam_meta,
        )

        if self.config.downscale_factor is not None:
            cameras.rescale_output_resolution(
                scaling_factor=1.0 / self.config.downscale_factor,
                scale_rounding_mode=self.config.downscale_rounding_mode,
            )

        if not self.config.read_exr and len(image_filenames) > 0:
            # Get bayer mask if we train on raw pixels.
            image_coords = cameras.get_image_coords(pixel_offset=0)
            bayer_mask = raw_utils.pixels_to_bayer_mask(
                image_coords[..., 1], image_coords[..., 0], bayer_pattern[0].tolist()
            )
            metadata.update({"bayer_mask": bayer_mask})

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        if self.config.load_3D_points:
            # Load 3D points
            metadata.update(self._load_3D_points(colmap_path, transform_matrix, scale_factor))

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": (depth_filenames if len(depth_filenames) > 0 else None),
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )
        return dataparser_outputs
