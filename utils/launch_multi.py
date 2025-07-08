import argparse
import os
import subprocess
import glob
from natsort import natsorted
from typing import Literal
from dataclasses import dataclass
from imgs_to_vid import imgs_to_vid


def run_relative_illumination(config_path: str):
    @dataclass
    class Config:
        """Absolute path to the dataset directory."""

        path: str
        """Dataset type, choose from ['synthetic-medium', 'synthetic-air', 'real-medium', 'real-air']."""
        type: Literal["synthetic-medium", "synthetic-air", "real-medium", "real-air"]
        """Experiment name."""
        experiment_name: str
        """Skip this dataset if True."""
        skip: bool
        """Model parameter: alpha_scale."""
        alpha_scale: float

    def load_run_configs(file_path: str) -> list[Config]:
        """Load run configurations from a YAML file."""
        import yaml

        with open(file_path, "r") as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        return [Config(**config) for config in yaml_data["configs"]]

    def run_dataset(config: Config):
        method_name = "relative-illumination" if "synthetic" in config.type else "relative-illumination-big"
        # Training command
        train_cmds = [
            "ns-train",
            method_name,
            "--data",
            config.path,
            "--output-dir",
            os.path.join(config.path, "results"),
            "--experiment-name",
            config.experiment_name,
            "--pipeline.model.model_medium",
            "True" if "medium" in config.type else "False",
            "--pipeline.model.alpha_scale",
            f"{config.alpha_scale}",
            "--pipeline.model.apply_srgb",
            "True",
            "--vis",
            "tensorboard",
        ]
        if "synthetic" in config.type:
            train_cmds += [
                "colmap-raw",
                "--raw_path",
                "images",
                "--disable-raw",
                "False",
                "--read-exr",
                "True",
                "--load_3D_points",
                "False",
            ]
        subprocess.run(train_cmds)

        # Eval command
        result_path = os.path.join(config.path, "results", config.experiment_name, method_name)
        result_path = sorted(glob.glob(result_path + "/*"))[-1]
        eval_cmds = [
            "ns-eval",
            "--load-config",
            os.path.join(result_path, "config.yml"),
            "--output-path",
            os.path.join(result_path, "metric.json"),
        ]
        subprocess.run(eval_cmds)

        # Render command
        render_cmds = [
            "ns-render",
            "dataset",
            "--load-config",
            os.path.join(result_path, "config.yml"),
            "--output-path",
            os.path.join(result_path, "render"),
            "--split",
            "train+test",
            "--rendered-output-names",
        ]
        if "medium" in config.type:
            render_cmds += ["rgb", "rgb_clean", "depth", "alpha_map", "rgb_medium"]
        elif "air" in config.type:
            render_cmds += ["rgb", "rgb_clean", "depth", "alpha_map"]
        subprocess.run(render_cmds)

        # Compose video using ffmpeg
        if "medium" in config.type:
            renders = ["rgb", "rgb_clean", "depth", "alpha_map", "rgb_medium"]
        elif "air" in config.type:
            renders = ["rgb", "rgb_clean", "depth", "alpha_map"]
        mode = ["train", "test"]
        for render in renders:
            for m in mode:
                render_path = os.path.join(result_path, "render", f"{m}", render)
                video_path = os.path.join(result_path, "render", f"{m}", f"{m}_{render}.mp4")
                imgs_to_vid(render_path, video_path)

    configs = load_run_configs(config_path)
    for config in configs:
        if not config.skip:
            run_dataset(config)
    print("Finished all datasets for relative-illumination")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--run_configs",
        help="Path to the config file for running multiple experiments on different datasets",
        default="run_configs.yml",
        required=True,
    )
    parser.add_argument(
        "--method",
        help="Which method to run",
        choices=["relative-illumination"],
        required=True,
        default="relative-illumination",
    )
    args = parser.parse_args()

    if args.method == "relative-illumination":
        run_relative_illumination(args.run_configs)


if __name__ == "__main__":
    main()