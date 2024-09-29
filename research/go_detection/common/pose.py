import torch
import numpy as np


class Pose3d:
    def __init__(self, mat=None, device=None):
        if mat is None:
            mat = torch.eye(4, dtype=torch.float64)
        self.mat = mat.to(device)

    def __eq__(self, other):
        assert self.mat.device == other.mat.device
        assert self.mat.dtype == other.mat.dtype
        return torch.isclose(self.mat, other.mat).all()

    def __repr__(self):
        return f"Pose3d\n{repr(self.mat)}"

    def to(self, device):
        return Pose3d(self.mat, device=device)

    @property
    def device(self):
        return self.mat.device

    @property
    def dtype(self):
        return self.mat.dtype

    @property
    def xyz(self):
        return self.mat[:3, -1]

    @property
    def matrix(self):
        return self.mat

    def __matmul__(self, other):
        if isinstance(other, Pose3d):
            assert self.mat.device == other.mat.device
            assert self.mat.dtype == other.mat.dtype
            new_pose = Pose3d(torch.matmul(self.mat, other.mat))
            return new_pose
        elif isinstance(other, torch.Tensor):
            assert self.mat.device == other.device
            assert self.mat.dtype == other.dtype
            # Transforming a point
            new_point = Pose3d.transform_points(self, other)
            return new_point
        else:
            assert False

    def inverse(self):
        return Pose3d(self.mat.inverse())

    def compose(self, other):
        # Same as other @ self
        assert self.mat.device == other.mat.device
        assert self.mat.dtype == other.mat.dtype
        self.mat = torch.matmul(other.mat, self.mat)
        return self

    def translate(self, x=0, y=0, z=0):
        pose = Pose3d.from_translation(x, y, z)
        self.compose(pose)
        return self

    def rotate(self, roll=0, pitch=0, yaw=0):
        pose = Pose3d.from_euler_rotation(roll, pitch, yaw)
        self.compose(pose)
        return self

    def rotate_deg(self, roll=0, pitch=0, yaw=0):
        pose = Pose3d.from_euler_rotation_deg(roll, pitch, yaw)
        self.compose(pose)
        return self

    @staticmethod
    def transform_points(pose, points):
        # pose: Pose3d 4x4
        # points: N...x3 or N...x4
        # returns N...x4

        if points.shape[-1] == 3:
            fourth_dim = torch.ones(
                *points.shape[:-1], 1, dtype=points.dtype, device=points.device
            )
            points = torch.cat((points, fourth_dim), dim=-1)

        assert points.shape[-1] == 4

        transformed_points = torch.matmul(pose.mat, points.unsqueeze(-1)).squeeze(-1)
        return transformed_points

    @staticmethod
    def from_translation(x=0, y=0, z=0):
        mat = torch.tensor(
            [
                [1.0, 0.0, 0.0, x],
                [0.0, 1.0, 0.0, y],
                [0.0, 0.0, 1.0, z],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        return Pose3d(mat)

    @staticmethod
    def from_euler_rotation(roll=0, pitch=0, yaw=0):
        # x axis: roll
        # y axis: pitch
        # z axis: yaw

        alpha = yaw
        beta = pitch
        gamma = roll
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)

        mat = torch.tensor(
            [
                [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg, 0.0],
                [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg, 0.0],
                [-sb, cb * sg, cb * cg, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        return Pose3d(mat)

    @staticmethod
    def from_euler_rotation_deg(roll=0, pitch=0, yaw=0):
        return Pose3d.from_euler_rotation(
            np.radians(roll), np.radians(pitch), np.radians(yaw)
        )
