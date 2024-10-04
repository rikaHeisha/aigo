import itertools

import numpy as np
import torch
from go_detection.dataloader import DataPoint, DataPoints
from matplotlib import pyplot as plt


def _label_to_color(lab):
    if lab == 0:
        # black
        return "black"
    elif lab == 1:
        # empty
        return "#ECC350"
    else:
        # white
        assert lab == 2
        return "white"


def _draw_board(axis, board_grid_pt):
    colors = _label_to_color(1)
    axis.scatter(
        (board_grid_pt[:, 0]) + 0.5,
        (board_grid_pt[:, 1]) + 0.5,
        facecolors=colors,
        # edgecolors="#606060",
        # c=colors,
        marker="s",
        s=8000,
        # linewidth=3,
    )


def _draw_pieces(axis, grid_pt, label):
    label = label.reshape(-1)
    mask = (label != 1).nonzero().squeeze(1)

    pieces_grid_pt = grid_pt[mask]
    pieces_label = label[mask]
    assert (pieces_label != 1).all()

    pieces_colors = [_label_to_color(l) for l in pieces_label]
    axis.scatter(
        (pieces_grid_pt[:, 0]).int(),
        (pieces_grid_pt[:, 1]).int(),
        facecolors=pieces_colors,
        # edgecolors="#606060",
        # c=colors,
        marker="o",
        s=4000,
    )


def _draw_correct_incorrect(axis, grid_pt, label, predicted_label):
    colors = [("green" if l else "red") for l in (predicted_label == label).reshape(-1)]
    axis.scatter(
        (grid_pt[:, 0]).int(),
        (grid_pt[:, 1]).int(),
        facecolors=colors,
        # edgecolors="red",
        # c=colors,
        marker="o",
        s=4000,
    )


# TODO(rishi): change to 2x2 grid, and add original image to this. Fix the y flipped issue. Use GridSpec for exact grid layout
def visualize_grid(
    data_points: DataPoints,
    output_path: str,
    index: int,
    predicted_label: torch.Tensor,
):
    data_points = data_points.cpu()
    predicted_label = predicted_label.cpu()

    (num_images, _, height, width) = data_points.images.shape
    assert index < num_images
    label = data_points.labels[index]
    assert predicted_label.shape == label.shape

    xs = torch.arange(0, label.shape[0])
    ys = torch.arange(0, label.shape[1])
    meshgrid = torch.meshgrid(xs, ys, indexing="xy")
    grid_pt = torch.stack(meshgrid, dim=2).reshape(-1, 2)

    xs = torch.arange(0, label.shape[0] - 1)
    ys = torch.arange(0, label.shape[1] - 1)
    meshgrid = torch.meshgrid(xs, ys, indexing="xy")
    board_grid_pt = torch.stack(meshgrid, dim=2).reshape(-1, 2)

    # fig, axis = plt.subplots(figsize=(25, 25))
    fig, axes = plt.subplots(
        2, 2, gridspec_kw={"wspace": 0, "hspace": 0}, figsize=(25 * 3, 25 * 3)
    )
    if isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            axes = list(axes)
        else:
            axes = list(itertools.chain(*axes))
    else:
        axes = [axes]

    image = data_points.images[index]
    image = image.transpose(0, 1).transpose(1, 2)  # Convert CHW to HWC
    image = image.clamp(0.0, 1.0)

    axes[0].imshow(image)
    # _draw_board(axes[0], board_grid_pt)

    _draw_board(axes[1], board_grid_pt)
    _draw_pieces(axes[1], grid_pt, label)

    _draw_board(axes[2], board_grid_pt)
    _draw_pieces(axes[2], grid_pt, predicted_label)

    _draw_board(axes[3], board_grid_pt)
    _draw_correct_incorrect(axes[3], grid_pt, label, predicted_label)

    axes[0].axis("off")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(
        height,
    )

    # axes[0].set_facecolor("#222222")
    # axes[0].set_aspect("equal")
    # axes[0].set_xlim(-0.5, 18.5)
    # axes[0].set_ylim(-0.5, 18.5)
    # axes[0].xaxis.set_visible(False)  # Hide x-axis
    # axes[0].yaxis.set_visible(False)  # Hide y-axis
    # axes[0].set_xticklabels([])  # Hide x-axis labels
    # axes[0].set_yticklabels([])  # Hide y-axis labels
    # axes[0].spines["top"].set_visible(False)  # Hide the top spine
    # axes[0].spines["right"].set_visible(False)  # Hide the right spine
    # axes[0].spines["left"].set_visible(False)  # Hide the left spine
    # axes[0].spines["bottom"].set_visible(False)  # Hide the bottom spine

    for axis in axes[1:]:
        axis.set_facecolor("#222222")
        axis.set_aspect("equal")
        axis.set_xlim(-0.5, 18.5)
        axis.set_ylim(-0.5, 18.5)

        axis.xaxis.set_visible(False)  # Hide x-axis
        axis.yaxis.set_visible(False)  # Hide y-axis
        axis.set_xticklabels([])  # Hide x-axis labels
        axis.set_yticklabels([])  # Hide y-axis labels
        axis.spines["top"].set_visible(False)  # Hide the top spine
        axis.spines["right"].set_visible(False)  # Hide the right spine
        axis.spines["left"].set_visible(False)  # Hide the left spine
        axis.spines["bottom"].set_visible(False)  # Hide the bottom spine

        # axis.patch.set_edgecolor("white")
        # axis.patch.set_linewidth(10)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # plt.show()
    # sys.exit(0)


def visualize_grid_pyplot(
    data_points: DataPoints,
    output_path: str,
    index: int,
    predicted_label: torch.Tensor,
):
    pass


################################################################################################
# Old unused viz

# def _convert_points(points: torch.Tensor, board_pt: torch.Tensor):
#     """
#     Points is a tensor of (Nx2) points in normalized image coordinates (range [0,1]).
#     The points gets mapped to board_pt using bilinear interpolation. The board_pt contains the corners of the cube on the image
#     Returns:
#         A tensor of shape (Nx2) containing the mapped points (image points)
#     """
#     assert points.min() >= 0.0 and points.max() <= 1.0

#     # Perform lerp on the top edge
#     diff_x = board_pt[1] - board_pt[0]
#     points_top = board_pt[0] + points[:, 0].unsqueeze(1) * diff_x

#     # Perform lerp on the bottom edge
#     diff_x = board_pt[2] - board_pt[3]
#     points_bottom = board_pt[3] + points[:, 0].unsqueeze(1) * diff_x

#     # Perform lerp
#     diff_y = points_bottom - points_top
#     image_points = points_top + points[:, 1].unsqueeze(1) * diff_y

#     # # Perform lerp on the left edge
#     # diff_y = board_pt[3] - board_pt[0]
#     # points_left = board_pt[0] + points[:, 1].unsqueeze(1) * diff_y

#     # # Perform lerp on the right edge
#     # diff_y = board_pt[2] - board_pt[1]
#     # points_right = board_pt[1] + points[:, 1].unsqueeze(1) * diff_y

#     # # Perform lerp
#     # diff_x = points_right - points_left
#     # image_points_2 = points_left + points[:, 0].unsqueeze(1) * diff_x

#     return image_points

# def _visualize_single_helper(
#     axis,
#     image: torch.Tensor,
#     label: torch.Tensor,
#     board_pt: torch.Tensor,
#     viz_corner_points: bool = True,
#     viz_all_points: bool = False,
# ):
#     (_, height, width) = image.shape
#     image = image.transpose(0, 1).transpose(1, 2)  # Convert CHW to HWC
#     image = image.clamp(0.0, 1.0)

#     image = image.cpu()
#     label = label.cpu()
#     board_pt = board_pt.cpu()

#     axis.imshow(image)

#     # Corner points
#     if viz_corner_points:
#         axis.scatter(
#             (board_pt[:, 0] * width).int(),
#             (board_pt[:, 1] * height).int(),
#             # facecolors="none",
#             # edgecolors=["red", "red", "green", "green"],
#             c="red",
#             marker="s",
#             s=200,
#         )

#     if viz_all_points:
#         xs = torch.linspace(0.0, 1.0, steps=label.shape[0])
#         ys = torch.linspace(0.0, 1.0, steps=label.shape[1])
#         meshgrid = torch.meshgrid(xs, ys, indexing="xy")
#         grid_pt = torch.stack(meshgrid, dim=2).reshape(-1, 2)
#         image_points = _convert_points(grid_pt, board_pt)

#         colors = [_label_to_color(l) for l in label.reshape(-1)]

#         axis.scatter(
#             (image_points[:, 0] * width).int(),
#             (image_points[:, 1] * height).int(),
#             # facecolors="none",
#             # edgecolors=["red", "red", "green", "green"],
#             c=colors,
#             marker="s",
#             s=100,
#         )

#     axis.axis("off")
#     axis.set_aspect("auto")


# def visualize_single_datapoint(
#     data_points: DataPoints,
#     output_path: str,
#     index: int,
#     viz_corner_points: bool = True,
#     viz_all_points: bool = False,
# ):
#     num_images = data_points.images.shape[0]
#     assert index < num_images

#     fig, axis = plt.subplots(figsize=(25, 25))
#     _visualize_single_helper(
#         axis,
#         data_points.images[index],
#         data_points.labels[index],
#         data_points.board_pts[index],
#         viz_corner_points=viz_corner_points,
#         viz_all_points=viz_all_points,
#     )

#     # plt.show()
#     plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)


# def visualize_datapoints(
#     data_point: DataPoints,
#     output_path: str,
#     max_viz_images: int | None = None,
#     viz_corner_points: bool = True,
#     viz_all_points: bool = False,
# ):
#     def _get_layout(num_images: int):
#         known_layouts = [
#             (1, (1, 1)),
#             (2, (2, 1)),
#             (4, (2, 2)),
#             (6, (3, 2)),
#             (8, (2, 4)),
#             (9, (3, 3)),
#             (11, (3, 4)),
#             (12, (3, 4)),
#             (16, (4, 4)),
#         ]

#         for idx, layout in known_layouts:
#             if num_images <= idx:
#                 return layout

#         # Unknown layout. Return something
#         return (num_images, 1)

#     (num_images, _, height, width) = data_point.images.shape
#     if max_viz_images:
#         num_images = min(num_images, max_viz_images)

#     # gridspec_kw={"wspace": 0, "hspace": 0}
#     layout = _get_layout(num_images)
#     fig, axes = plt.subplots(
#         layout[0], layout[1], gridspec_kw={"wspace": 0, "hspace": 0}, figsize=(25, 25)
#     )
#     if isinstance(axes, np.ndarray):
#         if axes.ndim == 1:
#             axes = list(axes)
#         else:
#             axes = list(itertools.chain(*axes))
#     else:
#         axes = [axes]

#     for i in range(num_images):
#         _visualize_single_helper(
#             axes[i],
#             data_point.images[i],
#             data_point.labels[i],
#             data_point.board_pts[i],
#             viz_corner_points,
#             viz_all_points,
#         )

#     for i in range(num_images, len(axes)):
#         axes[i].axis("off")
#         axes[i].set_aspect("auto")

#     # plt.show()
#     plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)
