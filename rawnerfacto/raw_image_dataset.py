from typing import Dict, Literal

import numpy as np
import OpenEXR as exr
import rawpy
import torch
from jaxtyping import Float
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from torch import Tensor
from nerfstudio.utils.rich_utils import CONSOLE


class RawImageDataset(InputDataset):
    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, cache_compressed_images: bool = False
    ):
        super().__init__(dataparser_outputs, scale_factor)

    def get_image_raw(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        exposure_value = self._dataparser_outputs.cameras.metadata["exposure_values"][image_idx]

        with open(image_filename, "rb") as f:
            raw = rawpy.imread(f).raw_image

        def processing_fn(raw):
            raw = raw.astype(np.float32)
            black_level = self._dataparser_outputs.cameras.metadata["black_level"][image_idx].numpy()  # type: ignore
            white_level = self._dataparser_outputs.cameras.metadata["white_level"][image_idx].numpy()  # type: ignore
            im = (raw - black_level) / (white_level - black_level)

            # Currently raw data is always processed bayered which means that scaling must be disabled
            if self.scale_factor != 1.0:
                CONSOLE.print("[bold yellow]Warning: Use downscale_factor only during rendering!")
            im = torch.from_numpy(im.astype(np.float32)) / exposure_value
            im = im.unsqueeze(-1).expand(-1, -1, 3)
            return im

        im = processing_fn(raw)
        return im

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        read_exr = self._dataparser_outputs.image_filenames[image_idx].suffix.lower() == ".exr"
        read_dng = self._dataparser_outputs.image_filenames[image_idx].suffix.lower() == ".dng"

        if read_exr:
            image = exr.File(self._dataparser_outputs.image_filenames[image_idx].as_posix()).channels()["RGB"].pixels

            assert (
                image is not None
            ), f"Image {self._dataparser_outputs.image_filenames[image_idx]} could not be loaded!"
            assert image.dtype == np.float32, "EXR image not float32!"
            assert image.ndim == 3, "EXR image not three channel!"
            image = (image - self._dataparser_outputs.cameras.metadata["black_level"]) / (
                self._dataparser_outputs.cameras.metadata["white_level"]
                - self._dataparser_outputs.cameras.metadata["black_level"]
            )
            image = torch.tensor(image, dtype=torch.float32)
        elif read_dng:
            image = self.get_image_raw(image_idx)
        elif (
            self._dataparser_outputs.cameras.metadata is not None
            or "exposure" in self._dataparser_outputs.cameras.metadata
        ):
            # input_is_srgb, non raw, non linear, with exposure
            data = super().get_data(image_idx, image_type)
            exposure_value = self._dataparser_outputs.cameras.metadata["exposure_values"][image_idx]
            from rawnerfacto.raw_utils import srgb_to_linear

            im_linear = srgb_to_linear(data["image"])
            data["image"] = im_linear / exposure_value
            return data
        else:
            # no raw image
            return super().get_data(image_idx, image_type)

        data = {"image_idx": image_idx, "image": image}

        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0,
                data["image"],
                torch.ones_like(data["image"]) * torch.tensor(self.mask_color),
            )

        return data
