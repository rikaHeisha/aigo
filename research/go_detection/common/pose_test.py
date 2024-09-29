from common.pose import Pose3d
import torch
from math import pi


def test_translation():
    actual = Pose3d().translate(1.0, 2.0, 3.0).xyz
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    # torch.testing.assert_close(actual, expected)
    assert (actual == expected).all()


def test_rotation_translation():
    actual = Pose3d().rotate(0, 0, pi / 2).translate(1, 2, 3)
    expected = Pose3d(
        torch.tensor(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
    )
    assert actual == expected


def test_transformation():
    transformation = Pose3d().rotate_deg(yaw=90).translate(1, 0, 0)

    # Test 2 x 3
    points = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float64).reshape(2, 3)
    expected = torch.tensor([1, 1, 0, 1, 0, 0, 0, 1], dtype=torch.float64).reshape(2, 4)
    actual = transformation @ points
    assert torch.isclose(actual, expected).all()

    # Test 2 x 4
    points = torch.tensor([1, 0, 0, 1, 0, 1, 0, 1], dtype=torch.float64).reshape(2, 4)
    actual = transformation @ points
    assert torch.isclose(actual, expected).all()

    # Test 1 x 3
    points = torch.tensor([1, 0, 0], dtype=torch.float64).reshape(1, 3)
    expected = torch.tensor([1, 1, 0, 1], dtype=torch.float64).reshape(1, 4)
    actual = transformation @ points
    assert torch.isclose(actual, expected).all()


def test_matrix_multiplication():
    pose1 = Pose3d.from_translation(1, 2, 3)
    pose2 = Pose3d.from_euler_rotation_deg(0, 0, 90)

    actual = pose1 @ pose2
    expected = Pose3d().rotate_deg(0, 0, 90).translate(1, 2, 3)
    assert actual == expected

    actual = pose2 @ pose1
    expected = Pose3d().translate(1, 2, 3).rotate_deg(0, 0, 90)
    assert actual == expected
