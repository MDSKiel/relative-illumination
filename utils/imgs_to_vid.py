import os
import argparse
from natsort import natsorted
import subprocess
import tempfile
import glob


def imgs_to_vid(input_path, output_path):
    frame_paths = natsorted(glob.glob(os.path.join(input_path, "*")))

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as temp_file:
        for p in frame_paths:
            temp_file.write(f"file '{p}'\n")

    ffmpeg_cmds = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-r",
        str(10),
        "-i",
        temp_file.name,
        "-c:v",
        "libx264",
        "-crf",
        str(10),
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    subprocess.run(ffmpeg_cmds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="input images")
    parser.add_argument("output_path", help="output path to the video")
    args = parser.parse_args()

    imgs_to_vid(args.input_path, args.output_path)
