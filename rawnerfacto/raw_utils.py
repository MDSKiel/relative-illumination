from typing import Any, Mapping, MutableMapping, Optional, Sequence, List
from jaxtyping import Int
import numpy as np
import torch
from torch import Tensor

# The following code for raw data processing comes from RawNeRF:
# https://github.com/google-research/multinerf/blob/main/internal/raw_utils.py


def linear_to_srgb(linear: Tensor, eps: Optional[float] = None) -> Tensor:
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    eps = torch.tensor(eps).to(linear)
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.maximum(eps, linear) ** (5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb: Tensor, eps: Optional[float] = None) -> Tensor:
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    eps = torch.tensor(eps).to(srgb)
    linear0 = 25 / 323 * srgb
    linear1 = np.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
    return torch.where(srgb <= 0.04045, linear0, linear1)


def postprocess_raw(
    raw: Tensor, camtorgb: Tensor, exposure: Optional[float] = None, apply_srgb: Optional[bool] = True
) -> Tensor:
    """Converts demosaicked raw to sRGB with a minimal postprocessing pipeline.

    Args:
      raw: [H, W, 3], demosaicked raw camera image, must be in [0, 1]
      camtorgb: [3, 3], color correction transformation to apply to raw image.
      xnp: either numpy or jax.numpy.

    Returns:
      srgb: [H, W, 3], color corrected + exposed + gamma mapped image.
    """
    if raw.shape[-1] != 3:
        raise ValueError(f"raw.shape[-1] is {raw.shape[-1]}, expected 3")
    if camtorgb.shape != (3, 3):
        raise ValueError(f"camtorgb.shape is {camtorgb.shape}, expected (3, 3)")
    # Convert from camera color space to standard linear RGB color space.
    rgb_linear = torch.matmul(raw, camtorgb.T).to(raw)
    if exposure is None:
        exposure = torch.quantile(rgb_linear, 0.97)
    # "Expose" image by mapping the input exposure level to white and clipping.
    rgb_linear_scaled = torch.clip(rgb_linear / exposure, 0, 1)

    if apply_srgb:
        # Apply sRGB gamma curve to serve as a simple tonemap.
        return linear_to_srgb(rgb_linear_scaled)
    else:
        return rgb_linear_scaled


def bilinear_demosaic(bayer: np.ndarray, pattern: List[int]) -> np.ndarray:
    """Converts Bayer data into a full RGB image using bilinear demosaicking.

    Input data should be ndarray of shape [height, width] with 2x2 mosaic pattern:
      -------------
      |red  |green|
      -------------
      |green|blue |
      -------------
    Red and blue channels are bilinearly upsampled 2x, missing green channel
    elements are the average of the neighboring 4 values in a cross pattern.

    Args:
      bayer: [H, W] array, Bayer mosaic pattern input image.
      xnp: either numpy or jax.numpy.

      pattern: List of integers indicating the bayer pattern of the input array.

    Returns:
      rgb: [H, W, 3] array, full RGB image.
    """

    def reshape_quads(*planes):
        """Reshape pixels from four input images to make tiled 2x2 quads."""
        planes = np.stack(planes, -1)
        shape = planes.shape[:-1]
        # Create [2, 2] arrays out of 4 channels.
        zup = planes.reshape(
            shape
            + (
                2,
                2,
            )
        )
        # Transpose so that x-axis dimensions come before y-axis dimensions.
        zup = np.transpose(zup, (0, 2, 1, 3))
        # Reshape to 2D.
        zup = zup.reshape((shape[0] * 2, shape[1] * 2))
        return zup

    def bilinear_upsample(z):
        """2x bilinear image upsample."""
        # Using np.roll makes the right and bottom edges wrap around. The raw image
        # data has a few garbage columns/rows at the edges that must be discarded
        # anyway, so this does not matter in practice.
        # Horizontally interpolated values.
        zx = 0.5 * (z + np.roll(z, -1, axis=-1))
        # Vertically interpolated values.
        zy = 0.5 * (z + np.roll(z, -1, axis=-2))
        # Diagonally interpolated values.
        zxy = 0.5 * (zx + np.roll(zx, -1, axis=-2))
        return reshape_quads(z, zx, zy, zxy)

    def upsample_green(g1, g2):
        """Special 2x upsample from the two green channels."""
        z = np.zeros_like(g1)
        z = reshape_quads(z, g1, g2, z)
        alt = 0
        # Grab the 4 directly adjacent neighbors in a "cross" pattern.
        for i in range(4):
            axis = -1 - (i // 2)
            roll = -1 + 2 * (i % 2)
            alt = alt + 0.25 * np.roll(z, roll, axis=axis)
        # For observed pixels, alt = 0, and for unobserved pixels, alt = avg(cross),
        # so alt + z will have every pixel filled in.
        return alt + z

    if pattern == [0, 1, 1, 2]:
        # RGGB
        r, g1, g2, b = [bayer[(i // 2) :: 2, (i % 2) :: 2] for i in range(4)]
    elif pattern == [2, 1, 1, 0]:
        # BGGR
        b, g1, g2, r = [bayer[(i // 2) :: 2, (i % 2) :: 2] for i in range(4)]
    elif pattern == [1, 0, 2, 1]:
        # GRBG
        g1, r, b, g2 = [bayer[(i // 2) :: 2, (i % 2) :: 2] for i in range(4)]
    elif pattern == [1, 2, 0, 1]:
        # GBRG
        g1, b, r, g2 = [bayer[(i // 2) :: 2, (i % 2) :: 2] for i in range(4)]
    else:
        raise NotImplementedError(
            f"Unsupported Bayer pattern, please make sure the input data is one of 'RGGB', 'BGGR', 'GRBG', 'GBRG'"
        )

    r = bilinear_upsample(r)
    # Flip in x and y before and after calling upsample, as bilinear_upsample
    # assumes that the samples are at the top-left corner of the 2x2 sample.
    b = bilinear_upsample(b[::-1, ::-1])[::-1, ::-1]
    g = upsample_green(g1, g2)
    rgb = np.stack([r, g, b], -1)
    return rgb


# Relevant fields to extract from raw image EXIF metadata.
# For details regarding EXIF parameters, see:
# https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf.
_EXIF_KEYS = (
    "BlackLevel",  # Black level offset added to sensor measurements.
    "WhiteLevel",  # Maximum possible sensor measurement.
    "AsShotNeutral",  # RGB white balance coefficients.
    "ColorMatrix1",  # XYZ to camera color space conversion matrix under CalibrationIlluminant1.
    "ColorMatrix2",  # XYZ to camera color space conversion matrix under CalibrationIlluminant2.
    "NoiseProfile",  # Shot and read noise levels.
    "CFAPattern2",  # Bayer patttern of the input images.
)

# Color conversion from reference illuminant XYZ to RGB color space.
# See http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html.
_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


def process_exif(exifs: Sequence[Mapping[str, Any]]) -> MutableMapping[str, Any]:
    """Processes list of raw image EXIF data into useful metadata dict.

    Input should be a list of dictionaries loaded from JSON files.
    These JSON files are produced by running
      $ exiftool -json IMAGE.dng > IMAGE.json
    for each input raw file.

    We extract only the parameters relevant to
    1. Rescaling the raw data to [0, 1],
    2. White balance and color correction, and
    3. Noise level estimation.

    Args:
      exifs: a list of dicts containing EXIF data as loaded from JSON files.

    Returns:
      meta: a dict of the relevant metadata for running RawNeRF.
    """
    meta = {}
    exif = exifs[0]
    # Convert from array of dicts (exifs) to dict of arrays (meta).
    for key in _EXIF_KEYS:
        exif_value = exif.get(key)
        if exif_value is None:
            continue
        # Values can be a single int or float...
        if isinstance(exif_value, (float, int)):
            vals = [x[key] for x in exifs]
        # Or a string of numbers with ' ' between.
        elif isinstance(exif_value, str):
            vals = [[float(z) for z in x[key].split(" ")] for x in exifs]
        meta[key] = np.squeeze(np.array(vals))
    # Shutter speed is a special case, a string written like 1/N.
    meta["ShutterSpeed"] = np.fromiter((1.0 / float(exif["ShutterSpeed"].split("/")[1]) for exif in exifs), float)

    color_matrix_tag = "ColorMatrix2" if "ColorMatrix2" in meta else "ColorMatrix1"

    # Create raw-to-sRGB color transform matrices. Pipeline is:
    # cam space -> white balanced cam space ("camwb") -> XYZ space -> RGB space.
    # 'AsShotNeutral' is an RGB triplet representing how pure white would measure
    # on the sensor, so dividing by these numbers corrects the white balance.
    whitebalance = meta["AsShotNeutral"].reshape(-1, 3)
    cam2camwb = np.array([np.diag(1.0 / x) for x in whitebalance])
    # ColorMatrix2 converts from XYZ color space to "reference illuminant" (white
    # balanced) camera space.
    xyz2camwb = meta[color_matrix_tag].reshape(-1, 3, 3)
    rgb2camwb = xyz2camwb @ _RGB2XYZ
    # We normalize the rows of the full color correction matrix, as is done in
    # https://github.com/AbdoKamel/simple-camera-pipeline.
    rgb2camwb /= rgb2camwb.sum(axis=-1, keepdims=True)
    # Combining color correction with white balance gives the entire transform.
    cam2rgb = np.linalg.inv(rgb2camwb) @ cam2camwb
    meta["cam2rgb"] = cam2rgb
    meta["CFAPattern2"] = meta["CFAPattern2"].astype(np.int32)
    return meta


def downsample(img, factor):
    """Area downsample img (factor must evenly divide img height and width)."""
    sh = img.shape
    if not (sh[0] % factor == 0 and sh[1] % factor == 0):
        raise ValueError(f"Downsampling factor {factor} does not evenly divide image shape {sh[:2]}")
    img = img.reshape((sh[0] // factor, factor, sh[1] // factor, factor) + sh[2:])
    img = img.mean((1, 3))
    return img


def pixels_to_bayer_mask(pix_x: Tensor, pix_y: Tensor, pattern: List[int]) -> Tensor:
    """Computes binary RGB Bayer mask values from integer pixel coordinates."""
    if pattern == [0, 1, 1, 2]:
        # RGGB
        # Red is top left (0, 0).
        r = (pix_x % 2 == 0) * (pix_y % 2 == 0)
        # Green is top right (0, 1) and bottom left (1, 0).
        g = (pix_x % 2 == 1) * (pix_y % 2 == 0) + (pix_x % 2 == 0) * (pix_y % 2 == 1)
        # Blue is bottom right (1, 1).
        b = (pix_x % 2 == 1) * (pix_y % 2 == 1)
    elif pattern == [2, 1, 1, 0]:
        # BGGR
        # Blue is top left (0, 0).
        b = (pix_x % 2 == 0) * (pix_y % 2 == 0)
        # Green is top right (0, 1) and bottom left (1, 0).
        g = (pix_x % 2 == 1) * (pix_y % 2 == 0) + (pix_x % 2 == 0) * (pix_y % 2 == 1)
        # Red is bottom right (1, 1).
        r = (pix_x % 2 == 1) * (pix_y % 2 == 1)
    elif pattern == [1, 0, 2, 1]:
        # GRBG
        # Green is top left (0, 0) and bottom right (1, 1).
        g = (pix_x % 2 == 0) * (pix_y % 2 == 0) + (pix_x % 2 == 1) * (pix_y % 2 == 1)
        # Red is top right (0, 1)
        r = (pix_x % 2 == 0) * (pix_y % 2 == 1)
        # Blue is bottom left (1, 0)
        b = (pix_x % 2 == 1) * (pix_y % 2 == 0)
    elif pattern == [1, 2, 0, 1]:
        # GBRG
        # Green is top left (0, 0) and bottom right (1, 1).
        g = (pix_x % 2 == 0) * (pix_y % 2 == 0) + (pix_x % 2 == 1) * (pix_y % 2 == 1)
        # Blue is top right (0, 1)
        b = (pix_x % 2 == 0) * (pix_y % 2 == 1)
        # Red is bottom left (1, 0)
        r = (pix_x % 2 == 1) * (pix_y % 2 == 0)

    return torch.stack([r, g, b], -1).to(torch.float32)