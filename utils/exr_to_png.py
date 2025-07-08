import argparse
import glob
import os
import cv2
import numpy as np
import tqdm


def convert_to_uint8_png(in_path, out_path, image_file_extension, multiplier, as16bit, percentile, apply_srgb):
    img_paths = glob.glob(os.path.join(in_path, "*." + image_file_extension))

    imgs = []
    max = 0
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for file_path in tqdm.tqdm(img_paths):
        outp = os.path.splitext(os.path.join(out_path, os.path.basename(file_path)))[0] + ".png"
        if file_path.endswith("exr"):
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        img = cv2.imread(
            file_path,
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )

        imgs.append((img, outp))
        img_max = np.percentile(img, float(percentile))
        if img_max > max:
            max = img_max

    iinfo = np.iinfo(np.uint16) if as16bit else np.iinfo(np.uint8)

    for img, outp in imgs:
        img = np.clip(img, 0, max)
        img = img / max

        if multiplier != 1:
            img = np.clip(float(multiplier) * img, 0, 1)

        if apply_srgb:
            eps = np.finfo(np.float32).eps
            srgb0 = 323 / 25 * img
            srgb1 = (211 * np.maximum(eps, img) ** (5 / 12) - 11) / 200
            img = np.where(img <= 0.0031308, srgb0, srgb1)

        img = (img * iinfo.max).astype(iinfo.dtype)
        cv2.imwrite(outp, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_dir", help="input images directory")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("--image_file_extension", "-e", default="exr")
    parser.add_argument("--multiplier", "-m", default=1.0)
    parser.add_argument("--bit16", "-b", default=False, action="store_true")
    parser.add_argument("--percentile", "-p", default=100)
    parser.add_argument("--apply_srgb", help="apply sRGB to the images", default=False, action="store_true")

    args = parser.parse_args()

    convert_to_uint8_png(
        args.input_dir,
        args.output_dir,
        args.image_file_extension,
        args.multiplier,
        args.bit16,
        args.percentile,
        args.apply_srgb,
    )
