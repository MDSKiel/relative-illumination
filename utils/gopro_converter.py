import rawpy
import json
import argparse
import glob
import os
import subprocess
from tqdm import tqdm
import numpy as np
import torch
from imageio import imwrite
from rawnerfacto.raw_utils import process_exif, postprocess_raw, bilinear_demosaic

_GPR_TOOL = "gpr_tools"


def gopro2dng(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    paths = glob.glob(os.path.join(input_dir, "*.GPR"))
    for p in tqdm(paths):
        basename = os.path.splitext(os.path.basename(p))[0]
        dng_path = os.path.join(output_dir, basename + ".DNG")
        json_path = os.path.join(output_dir, basename + ".json")
        cmds = [_GPR_TOOL, "-i", p, "-o", dng_path]
        subprocess.run(cmds)
        os.system(f"exiftool -json {dng_path} > {json_path}")


def extract_metadata(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_paths = glob.glob(os.path.join(input_dir, "*"))
    paths = []
    for p in all_paths:
        if os.path.splitext(p)[1].lower() == ".dng":
            paths.append(p)
    for p in tqdm(paths):
        basename = os.path.splitext(os.path.basename(p))[0]
        json_path = os.path.join(output_dir, basename + ".json")
        os.system(f"exiftool -json {p} > {json_path}")


def post_process_raw(input_dir, output_dir, exposure_level=97.0, linear=False):
    os.makedirs(output_dir, exist_ok=True)
    all_paths = sorted(glob.glob(os.path.join(input_dir, "*")))
    paths = []
    for p in all_paths:
        if os.path.splitext(p)[1].lower() == ".dng":
            paths.append(p)

    def load_raw_exif(path):
        base = os.path.join(input_dir, os.path.splitext(os.path.basename(path))[0])
        with open(path, "rb") as f:
            raw = rawpy.imread(f).raw_image
        with open(base + ".json", "rb") as f:
            exif = json.load(f)[0]
        return raw, exif

    def processing_fn(raw, exif):
        raw = raw.astype(np.float32)
        meta = process_exif([exif])
        # Rescale raw sensor measurements to [0, 1] (plus noise).
        blacklevel = np.mean(meta["BlackLevel"])
        whitelevel = meta["WhiteLevel"]
        im = (raw - blacklevel) / (whitelevel - blacklevel)

        # Demosaic Bayer images (preserves the measured RGGB values).
        im = bilinear_demosaic(im, meta["CFAPattern2"].tolist())

        if not linear:
            cam2rgb = meta["cam2rgb"][0]
        else:
            cam2rgb = np.eye(3)
        im = postprocess_raw(
            torch.tensor(im, dtype=torch.float32),
            torch.tensor(cam2rgb, dtype=torch.float32),
            exposure=np.percentile(im, exposure_level),
            apply_srgb=False if linear else True,
        ).numpy()

        im = (np.clip(np.nan_to_num(im), 0.0, 1.0) * 255.0).astype(np.uint8)
        return im

    for path in tqdm(paths):
        raw, exif = load_raw_exif(path)
        im = processing_fn(raw, exif)
        imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0] + ".png"), im)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_dir", help="input raw images (.GPR / .DNG)")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument(
        "--gopro_to_dng",
        help="Convert raw GoPro .GPR format into .DNG format.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--extract_metadata",
        help="Extract metadata from already existing .DNG images.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--post_process_raw",
        help="Extract and post-process the raw .DNG files and convert them into .PNG format.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--exposure_level", help="Percentage exposure level to use for post-processing.", type=float, default=97.0
    )
    parser.add_argument(
        "--linear",
        help="Wheter to extract the raw images in linear intensity.",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if args.gopro_to_dng:
        gopro2dng(args.input_dir, args.output_dir)
    elif args.extract_metadata:
        extract_metadata(args.input_dir, args.output_dir)
    elif args.post_process_raw:
        post_process_raw(args.input_dir, args.output_dir, args.exposure_level, args.linear)


if __name__ == "__main__":
    main()