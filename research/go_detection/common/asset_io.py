import os
import pickle as pkl

from pathlib import Path
from typing import Dict
import torch
import open3d as o3d
from omegaconf import OmegaConf


class AssetIO:
    def __init__(self, base_path):
        self.base_path = base_path

    def _abs(self, rel_path: str):
        return os.path.join(self.base_path, rel_path)

    def get_abs(self, rel_path: str):
        return self._abs(rel_path)

    def mkdir(self, rel_path: str = ""):
        os.makedirs(self._abs(rel_path), exist_ok=True)

    def save_torch(self, rel_path: str, data):
        torch.save(
            data,
            self._abs(rel_path),
        )

    def load_torch(self, rel_path: str):
        data = torch.load(self._abs(rel_path))
        return data

    def save_yaml(self, rel_path: str, data):
        with open(self._abs(rel_path), mode="w") as fp:
            OmegaConf.save(config=data, f=fp.name)
            # fp.write(data)

    def load_yaml(self, rel_path: str):
        return OmegaConf.load(self._abs(rel_path))

    def save_pcd(
        self,
        rel_path: str,
        pcd: o3d.geometry.PointCloud,
        write_ascii: bool = True,
        print_progress: bool = False,
    ):
        success = o3d.io.write_point_cloud(
            self._abs(rel_path),
            pcd,
            write_ascii=write_ascii,
            print_progress=print_progress,
        )
        if not success:
            raise IOError(
                f"Could not save O3D point cloud to path {self._abs(rel_path)}"
            )

    def save_mesh(
        self,
        rel_path: str,
        mesh: o3d.geometry.TriangleMesh,
        write_ascii: bool = False,
        write_vertex_normals: bool = True,
        write_vertex_colors: bool = True,
        write_triangle_uvs: bool = True,
        print_progress: bool = False,
    ):
        success = o3d.io.write_triangle_mesh(
            self._abs(rel_path),
            mesh,
            write_ascii=write_ascii,
            write_vertex_normals=write_vertex_normals,
            write_vertex_colors=write_vertex_colors,
            write_triangle_uvs=write_triangle_uvs,
            print_progress=print_progress,
        )
        if not success:
            raise IOError(f"Could not save O3D mesh to path {self._abs(rel_path)}")
