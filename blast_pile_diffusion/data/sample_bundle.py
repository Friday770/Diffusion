"""一个样本的所有模态数据打包在一起，是整条流水线的统一数据接口。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class SampleBundle:
    scene_id: str
    cam_id: str

    # Unity 原始模态
    rgb: np.ndarray              # (H, W, 3) uint8
    depth: np.ndarray            # (H, W) float32, 米单位
    normal: np.ndarray           # (H, W, 3) float32, 相机空间单位法向量
    mask: np.ndarray             # (H, W) int32, 背景=0, 每个实例一个唯一 ID
    meta: dict = field(default_factory=dict)

    # 预处理后填充（ControlNet 输入格式）
    depth_cn: Optional[np.ndarray] = None   # (H, W, 3) uint8, 归一化伪彩色
    canny: Optional[np.ndarray] = None      # (H, W) uint8, 从 mask 派生
    normal_cn: Optional[np.ndarray] = None  # (H, W, 3) uint8, ControlNet 编码

    @property
    def sample_key(self) -> str:
        return f"{self.scene_id}--{self.cam_id}"

    @property
    def num_instances(self) -> int:
        ids = np.unique(self.mask)
        return int((ids > 0).sum())

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    def is_preprocessed(self) -> bool:
        return self.depth_cn is not None and self.canny is not None

    def save(self, out_dir: Path) -> None:
        """将 bundle 序列化到磁盘目录。"""
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
        np.save(out_dir / "depth.npy", self.depth)
        np.save(out_dir / "normal.npy", self.normal)
        cv2.imwrite(str(out_dir / "mask_instance.png"), self.mask.astype(np.uint16))

        if self.depth_cn is not None:
            cv2.imwrite(str(out_dir / "depth_cn.png"), self.depth_cn)
        if self.canny is not None:
            cv2.imwrite(str(out_dir / "canny_from_mask.png"), self.canny)
        if self.normal_cn is not None:
            cv2.imwrite(
                str(out_dir / "normal_cn.png"),
                cv2.cvtColor(self.normal_cn, cv2.COLOR_RGB2BGR),
            )

        with open(out_dir / "meta.json", "w") as f:
            json.dump({**self.meta, "num_instances": self.num_instances}, f, indent=2)

    @classmethod
    def load(cls, bundle_dir: Path) -> SampleBundle:
        """从磁盘目录反序列化。"""
        parts = bundle_dir.name.split("--", 1)
        scene_id = parts[0] if len(parts) > 0 else "unknown"
        cam_id = parts[1] if len(parts) > 1 else "cam0"

        rgb_bgr = cv2.imread(str(bundle_dir / "rgb.png"))
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(bundle_dir / "depth.npy")
        normal = np.load(bundle_dir / "normal.npy")
        mask = cv2.imread(str(bundle_dir / "mask_instance.png"), cv2.IMREAD_UNCHANGED).astype(
            np.int32
        )

        with open(bundle_dir / "meta.json") as f:
            meta = json.load(f)

        bundle = cls(
            scene_id=scene_id,
            cam_id=cam_id,
            rgb=rgb,
            depth=depth,
            normal=normal,
            mask=mask,
            meta=meta,
        )

        depth_cn_path = bundle_dir / "depth_cn.png"
        if depth_cn_path.exists():
            bundle.depth_cn = cv2.imread(str(depth_cn_path))
        canny_path = bundle_dir / "canny_from_mask.png"
        if canny_path.exists():
            bundle.canny = cv2.imread(str(canny_path), cv2.IMREAD_GRAYSCALE)
        normal_cn_path = bundle_dir / "normal_cn.png"
        if normal_cn_path.exists():
            bundle.normal_cn = cv2.cvtColor(
                cv2.imread(str(normal_cn_path)), cv2.COLOR_BGR2RGB
            )

        return bundle
