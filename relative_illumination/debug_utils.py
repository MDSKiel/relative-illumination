import os
import re
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh


def natsort(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split("(\\d+)", s)]


def query_illumination_field(
    illumination_field, device, x_lims=(-0.75, 0.75), y_lims=(-0.75, 0.75), step_size=0.01, z_lim: int = 1
) -> Tuple[List[np.ndarray]]:
    with torch.no_grad():

        class frustrum_mock:
            def __init__(self, z_slice, device):
                lower_x, upper_x = x_lims
                lower_y, upper_y = y_lims
                X, Y = np.mgrid[lower_x:upper_x:step_size, lower_y:upper_y:step_size]
                Z = np.full_like(X, z_slice)
                self.p = (
                    torch.tensor(np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T.astype(np.float32))
                    .to(device)
                    .unsqueeze(1)
                )
                self.shape = self.p.shape[:-1]

            def get_positions(self):
                return self.p

        class ray_sample_mock:
            def __init__(self, device, z_slice) -> None:
                self.frustums = frustrum_mock(z_slice, device)
                metadata = {
                    "c2ws": torch.eye(3, 4, device=device)
                    .unsqueeze(0)
                    .expand((self.frustums.shape[0], 3, 4))
                    .unsqueeze(1)
                }
                self.metadata = metadata
                self.device = device

        colors = []
        vertices = []
        for z_slice in np.linspace(start=0, stop=z_lim, num=int(z_lim / step_size)):
            mock = ray_sample_mock(device, z_slice)
            illumination_field_out = illumination_field.forward(mock, None)["illumination_field"]
            colors.append(illumination_field_out.cpu().numpy())
            vertices.append(mock.frustums.get_positions().cpu().numpy())

        return (colors, vertices)


def visualize_illumination_field_as_mesh(
    illumination_field,
    output_file: str,
    device,
    x_lims=(-0.75, 0.75),
    y_lims=(-0.75, 0.75),
    step_size=0.01,
    z_lim: int = 1,
):
    colors, vertices = query_illumination_field(illumination_field, device, x_lims, y_lims, step_size, z_lim)

    colors = np.array(colors).ravel().reshape(-1, 3)
    vertices = np.array(vertices).ravel().reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, vertex_colors=(colors * 255).astype(np.uint8))
    mesh.export(output_file)


def visualize_illumination_field_as_images(
    illumination_field,
    output_dir: str,
    device,
    x_lims=(-0.75, 0.75),
    y_lims=(-0.75, 0.75),
    step_size=0.01,
    z_lim: int = 1,
):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    colors, vertices = query_illumination_field(illumination_field, device, x_lims, y_lims, step_size, z_lim)

    lower_x, upper_x = x_lims
    lower_y, upper_y = y_lims
    X, _ = np.mgrid[lower_x:upper_x:step_size, lower_y:upper_y:step_size]
    imshape = X.shape + (colors[0].shape[-1],)

    for i, img in enumerate(colors):
        img = (img.reshape(imshape) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), img)  # is nerfstudio also BGR? Results suggest it is.


def create_graph_from_imgs(dir: str, aoi_topleft_bottomright=(60, 60, 90, 90)):
    r = []
    g = []
    b = []
    files = sorted(os.listdir(dir), key=natsort)
    for file in files:
        img = cv2.imread(os.path.join(dir, file))
        top, left, bottom, right = aoi_topleft_bottomright
        crop = img[top:bottom, left:right, :] / 255
        b.append(np.mean(crop[..., 0]))
        g.append(np.mean(crop[..., 1]))
        r.append(np.mean(crop[..., 2]))

    x = range(len(r))

    fig, ax = plt.subplots()
    ax.plot(x, r, "r")
    ax.plot(x, g, "g")
    ax.plot(x, b, "b")

    ax.grid()
    plt.show()
    plt.waitforbuttonpress()
