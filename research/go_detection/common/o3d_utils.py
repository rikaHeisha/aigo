import numpy as np
import torch
import open3d as o3d

def np_to_o3d_pcd(points: np.ndarray, colors: np.ndarray = None, normals: np.ndarray = None):
    # colors can either be (3,) or N x 3
    assert points.ndim == 2 and points.shape[1] == 3
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    
    if colors is not None:
        if colors.shape == (3,):
            pcd.paint_uniform_color(colors)
        else:
            assert points.shape == colors.shape
            pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors))

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd