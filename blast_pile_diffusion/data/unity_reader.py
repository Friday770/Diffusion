"""解析 Unity Perception 输出目录结构，生成 SampleBundle 列表。

Unity Perception 的导出格式在不同包版本之间差异较大：有的场景把
captures JSON 放在 ``captures/`` 子目录，有的直接放在场景根目录；图像
路径也可能只写在 JSON 里，或者按 RGB / Depth / InstanceSegmentation
文件夹存放。本模块优先使用 metadata 中的路径和 colormap，缺失时再按
目录结构兜底扫描。
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from blast_pile_diffusion.data.sample_bundle import SampleBundle
from blast_pile_diffusion.utils.image_io import (
    decode_instance_mask_rgb,
    read_depth,
    read_normal_unity,
)

# 私有别名：保留兼容，避免历史调用点崩溃。新代码请直接用 image_io 公共 API。
_read_normal_unity = read_normal_unity
_decode_instance_mask_rgb = decode_instance_mask_rgb

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".exr", ".tiff", ".tif"}
ARRAY_SUFFIXES = {".npy"}
READABLE_SUFFIXES = IMAGE_SUFFIXES | ARRAY_SUFFIXES
FrameKey = tuple[int, str]


def find_unity_scenes(unity_raw_dir: Path) -> list[Path]:
    """查找 Unity Perception 场景目录。

    支持三种常见布局：
    1. ``scene/captures/*.json``；
    2. ``scene/captures_000.json``；
    3. 只有 RGB/Depth 等模态文件夹、没有 metadata 的合成 fixture。
    """
    unity_raw_dir = Path(unity_raw_dir)
    if not unity_raw_dir.exists():
        return []

    candidates: set[Path] = set()
    if _is_scene_dir(unity_raw_dir):
        candidates.add(unity_raw_dir)

    for captures_dir in unity_raw_dir.rglob("captures"):
        if captures_dir.is_dir():
            candidates.add(captures_dir.parent)

    for metadata_path in unity_raw_dir.rglob("*.json"):
        if _looks_like_capture_metadata(metadata_path):
            scene_dir = metadata_path.parent.parent if metadata_path.parent.name == "captures" else metadata_path.parent
            candidates.add(scene_dir)

    for directory in unity_raw_dir.rglob("*"):
        if directory.is_dir() and _has_modality_subdirs(directory):
            candidates.add(directory)

    return sorted(candidates)


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", name).lower()


def _is_scene_dir(path: Path) -> bool:
    return _metadata_files(path) or _has_modality_subdirs(path)


def _looks_like_capture_metadata(path: Path) -> bool:
    name = _norm_name(path.name)
    return "capture" in name or path.parent.name == "captures"


def _has_modality_subdirs(path: Path) -> bool:
    if not path.is_dir():
        return False
    modalities = {_classify_modality(child, child.name) for child in path.iterdir() if child.is_dir()}
    return "rgb" in modalities and "depth" in modalities


def _metadata_files(scene_dir: Path) -> list[Path]:
    paths: set[Path] = set()
    captures_dir = scene_dir / "captures"
    if captures_dir.exists():
        paths.update(p for p in captures_dir.glob("*.json") if p.is_file())
    paths.update(p for p in scene_dir.glob("*.json") if p.is_file() and _looks_like_capture_metadata(p))
    return sorted(paths)


def _classify_modality(path: Path | str, hint: str = "") -> str | None:
    text = _norm_name(f"{hint} {Path(path).as_posix()}")
    if "instancesegmentation" in text or "instancemask" in text:
        return "mask"
    if "instance" in text and ("segmentation" in text or "mask" in text):
        return "mask"
    if "segmentation" in text and "semantic" not in text:
        return "mask"
    if "normal" in text:
        return "normal"
    if "depth" in text or "distance" in text:
        return "depth"
    if "rgb" in text or "color" in text or "image" in text or "camera" in text:
        return "rgb"
    return None


def _modality_from_prefix(subfolder_prefix: str) -> str:
    classified = _classify_modality(subfolder_prefix, subfolder_prefix)
    return classified or _norm_name(subfolder_prefix)


def _list_frames(scene_dir: Path, subfolder_prefix: str) -> dict[FrameKey, Path]:
    frames: dict[FrameKey, Path] = {}
    modality = _modality_from_prefix(subfolder_prefix)
    if not scene_dir.exists():
        return frames

    for path in sorted(scene_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in READABLE_SUFFIXES:
            continue
        if _classify_modality(path, " ".join(path.parts)) != modality:
            continue
        frame_id = _extract_frame_id(path.name)
        if frame_id is not None:
            frames[(frame_id, "cam")] = path
    return frames


def _extract_frame_id(filename: str) -> int | None:
    stem = re.sub(r"\s+", "", Path(str(filename)).stem)
    match = re.search(r"_(\d+)$", stem)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", stem)
    return int(numbers[-1]) if numbers else None


def _color_to_rgb(color: Any) -> tuple[int, int, int] | None:
    if isinstance(color, dict):
        keys = ("r", "g", "b")
        if all(key in color for key in keys):
            color = [color[key] for key in keys]
        elif all(key.upper() in color for key in keys):
            color = [color[key.upper()] for key in keys]
        else:
            return None
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return None
    values = list(color[:3])
    try:
        if any(isinstance(v, float) for v in values) and all(0 <= float(v) <= 1 for v in values):
            values = [round(float(v) * 255) for v in values]
        rgb = tuple(int(round(float(v))) for v in values)
    except (TypeError, ValueError):
        return None
    return rgb if all(0 <= v <= 255 for v in rgb) else None


def _extract_colormap(metadata_dicts: list[dict]) -> dict[tuple[int, int, int], int]:
    colormap: dict[tuple[int, int, int], int] = {}
    stack: list[Any] = list(metadata_dicts)
    instance_keys = ("instance_id", "instanceId", "instanceID", "object_id", "objectId", "id")
    color_keys = ("color", "pixel_value", "pixelValue", "rgb", "rgba", "value")

    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            instance_id = next((obj.get(key) for key in instance_keys if key in obj), None)
            color = next((obj.get(key) for key in color_keys if key in obj), None)
            if isinstance(instance_id, str) and instance_id.isdigit():
                instance_id = int(instance_id)
            if isinstance(instance_id, int) and color is not None:
                rgb = _color_to_rgb(color)
                if rgb is not None and rgb != (0, 0, 0):
                    colormap[rgb] = int(instance_id)
            stack.extend(obj.values())
        elif isinstance(obj, list):
            stack.extend(obj)
    if not colormap:
        warnings.warn("未从 Unity 元数据中提取到 InstanceSegmentation colormap")
    return colormap


def _metadata_frame_id(capture: dict) -> int | None:
    for key in ("step", "frame", "sequence", "sequence_id", "sequenceId"):
        value = capture.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    for value in (capture.get("filename"), capture.get("id")):
        if isinstance(value, str) and (frame_id := _extract_frame_id(value)) is not None:
            return frame_id
    return None


def _metadata_camera_id(capture: dict) -> str:
    sensor = capture.get("sensor")
    if isinstance(sensor, dict):
        for key in ("sensor_id", "sensorId", "id", "name"):
            value = sensor.get(key)
            if isinstance(value, str) and value:
                return _safe_id(value)
    for key in ("camera", "camera_id", "cameraId"):
        value = capture.get(key)
        if isinstance(value, str) and value:
            return _safe_id(value)
    return "cam"


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip())
    return safe.strip("-") or "cam"


def _metadata_frame_key(capture: dict, fallback_index: int) -> FrameKey:
    frame_id = _metadata_frame_id(capture)
    if frame_id is None:
        frame_id = fallback_index
    return frame_id, _metadata_camera_id(capture)

def _capture_entries(metadata: Any) -> list[dict]:
    if isinstance(metadata, dict):
        captures = metadata.get("captures")
        if isinstance(captures, list):
            return [item for item in captures if isinstance(item, dict)]
        keys = ("step", "frame", "sequence", "filename", "id", "annotations")
        return [metadata] if any(key in metadata for key in keys) else []
    return [item for item in metadata if isinstance(item, dict)] if isinstance(metadata, list) else []


def _merge_frame_meta(existing: dict, update: dict) -> dict:
    merged = dict(existing)
    for key, value in update.items():
        if key in merged and merged[key] != value:
            if not isinstance(merged[key], list):
                merged[key] = [merged[key]]
            merged[key].append(value)
        else:
            merged[key] = value
    return merged


def _string_hint(obj: dict, fallback: str = "") -> str:
    hint_parts = [fallback]
    for key in ("id", "name", "description", "type", "annotation_definition", "labeler_id"):
        value = obj.get(key)
        if isinstance(value, (str, int, float)):
            hint_parts.append(str(value))
        elif isinstance(value, dict):
            hint_parts.extend(str(v) for v in value.values() if isinstance(v, (str, int, float)))
    return " ".join(hint_parts)


def _resolve_path(scene_dir: Path, raw_path: str) -> Path | None:
    raw_path = raw_path.strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.suffix.lower() not in READABLE_SUFFIXES:
        return None
    if path.is_absolute():
        return path if path.exists() else None

    candidates = [
        scene_dir / path,
        scene_dir / "captures" / path,
        scene_dir.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(scene_dir.rglob(path.name))
    return matches[0] if matches else candidates[0]


def _collect_path_refs(obj: Any, hint: str = "") -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        local_hint = _string_hint(obj, hint)
        for key, value in obj.items():
            key_hint = f"{local_hint} {key}"
            if key.lower() in {"filename", "file_name", "path", "file", "image"}:
                if isinstance(value, str):
                    refs.append((value, key_hint))
                elif isinstance(value, dict):
                    refs.extend(_collect_path_refs(value, key_hint))
            elif isinstance(value, (dict, list)):
                refs.extend(_collect_path_refs(value, key_hint))
    elif isinstance(obj, list):
        for item in obj:
            refs.extend(_collect_path_refs(item, hint))
    return refs


def _capture_paths(scene_dir: Path, capture: dict) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    capture_filename = capture.get("filename")
    if isinstance(capture_filename, str):
        resolved = _resolve_path(scene_dir, capture_filename)
        if resolved is not None:
            paths[_classify_modality(resolved, capture_filename) or "rgb"] = resolved

    for annotation in capture.get("annotations", []) if isinstance(capture.get("annotations"), list) else []:
        if not isinstance(annotation, dict):
            continue
        for raw_path, hint in _collect_path_refs(annotation, _string_hint(annotation)):
            resolved = _resolve_path(scene_dir, raw_path)
            if resolved is None:
                continue
            modality = _classify_modality(resolved, hint)
            if modality is not None:
                paths[modality] = resolved

    return paths


def _load_metadata(scene_dir: Path) -> dict:
    result: dict[str, Any] = {
        "frames": {},
        "paths": {"rgb": {}, "depth": {}, "normal": {}, "mask": {}},
        "colormap": {},
        "raw": [],
    }
    metadata_paths = _metadata_files(scene_dir)
    if not metadata_paths:
        return result

    capture_index = 0
    for path in metadata_paths:
        try:
            with open(path) as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(f"跳过无法读取的 Unity metadata: {path} ({exc})")
            continue
        if not isinstance(metadata, dict):
            continue
        result["raw"].append(metadata)
        for capture in _capture_entries(metadata):
            frame_key = _metadata_frame_key(capture, capture_index)
            capture_index += 1
            frames = result["frames"]
            frames[frame_key] = _merge_frame_meta(frames.get(frame_key, {}), capture)
            for modality, modality_path in _capture_paths(scene_dir, capture).items():
                result["paths"][modality][frame_key] = modality_path
    result["colormap"] = _extract_colormap(result["raw"])
    return result


def _read_rgb(path: Path) -> np.ndarray:
    rgb_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"无法读取 RGB: {path}")
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)


def _read_depth_meters(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = read_depth(path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32, copy=False)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def _read_mask(path: Path, colormap: dict[tuple[int, int, int], int]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
        return mask.astype(np.int32)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取 InstanceSegmentation: {path}")
    if img.ndim == 2:
        return img.astype(np.int32)
    mask_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    return _decode_instance_mask_rgb(mask_rgb, colormap)


def _resize_to_shape(array: np.ndarray, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    if array.shape[:2] == shape:
        return array
    resized = cv2.resize(array, (shape[1], shape[0]), interpolation=interpolation)
    return resized.astype(array.dtype, copy=False)


def _bundle_cam_id(frame_key: FrameKey) -> str:
    frame_id, camera_id = frame_key
    return f"cam{frame_id}" if camera_id == "cam" else f"{camera_id}_f{frame_id:06d}"

def iter_bundles(scene_dir: Path) -> Iterator[SampleBundle]:
    metadata = _load_metadata(scene_dir)
    frame_meta = metadata["frames"]
    metadata_paths = metadata["paths"]
    colormap = metadata["colormap"]
    scene_id = scene_dir.name
    rgb_frames = metadata_paths["rgb"] or _list_frames(scene_dir, "RGB")
    depth_frames = metadata_paths["depth"] or _list_frames(scene_dir, "Depth")
    normal_frames = metadata_paths["normal"] or _list_frames(scene_dir, "Normal")
    mask_frames = metadata_paths["mask"] or _list_frames(scene_dir, "InstanceSegmentation")
    warned_empty_colormap = False

    required_keys = set(rgb_frames) & set(depth_frames)
    for frame_key in sorted((set(rgb_frames) | set(depth_frames)) - required_keys):
        warnings.warn(f"跳过缺少 RGB 或 Depth 的 Unity 帧: {frame_key}")
    for frame_key in sorted(required_keys):
        try:
            rgb = _read_rgb(rgb_frames[frame_key])
        except (OSError, ValueError) as exc:
            warnings.warn(f"跳过无法读取 RGB 的 Unity 帧: {frame_key} ({exc})")
            continue
        shape = rgb.shape[:2]
        try:
            depth = _read_depth_meters(depth_frames[frame_key])
        except (OSError, ValueError) as exc:
            warnings.warn(f"跳过无法读取 Depth 的 Unity 帧: {frame_key} ({exc})")
            continue
        depth = _resize_to_shape(depth, shape, cv2.INTER_LINEAR).astype(np.float32)

        normal = np.zeros((*shape, 3), dtype=np.float32)
        if frame_key in normal_frames:
            try:
                normal = _read_normal_unity(normal_frames[frame_key])
                normal = _resize_to_shape(normal, shape, cv2.INTER_LINEAR).astype(np.float32)
            except (OSError, ValueError) as exc:
                warnings.warn(f"Unity 帧 {frame_key} 法线读取失败，使用零法线: {exc}")
        else:
            warnings.warn(f"Unity 帧 {frame_key} 缺少 Normal，使用零法线")

        mask = np.zeros(shape, dtype=np.int32)
        if frame_key in mask_frames:
            try:
                if not colormap and not warned_empty_colormap:
                    warnings.warn("未找到 colormap，将按非黑 RGB 颜色自动分配实例 ID")
                    warned_empty_colormap = True
                mask = _read_mask(mask_frames[frame_key], colormap)
                mask = _resize_to_shape(mask, shape, cv2.INTER_NEAREST).astype(np.int32)
                mask[mask < 0] = 0
            except (OSError, ValueError) as exc:
                warnings.warn(f"Unity 帧 {frame_key} 掩码读取失败，使用零掩码: {exc}")
        else:
            warnings.warn(f"Unity 帧 {frame_key} 缺少 InstanceSegmentation，使用零掩码")

        frame_id, _camera_id = frame_key
        yield SampleBundle(
            scene_id,
            _bundle_cam_id(frame_key),
            rgb.astype(np.uint8, copy=False),
            depth.astype(np.float32, copy=False),
            normal.astype(np.float32, copy=False),
            mask.astype(np.int32, copy=False),
            {"frame_id": frame_id, "camera_id": _camera_id, "raw": frame_meta.get(frame_key)},
        )
