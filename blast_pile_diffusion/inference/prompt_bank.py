"""Prompt 随机化模板 — 对应技术路线 §4.4.5。

每次调用 random_prompt() 组合不同的光照、天气、时段、场景细节，
确保生成图有足够多样性，防止 mode collapse。
"""

from __future__ import annotations

import hashlib
import random

TRIGGER_WORD = "<mine_blast_pile>"

LIGHTING = [
    "sunny noon",
    "overcast morning",
    "late afternoon golden hour",
    "dawn soft light",
    "harsh midday sun",
    "diffused cloudy light",
    "directional side light",
]

WEATHER = [
    "clear",
    "slight haze",
    "dusty",
    "after rain wet rocks",
    "windy dust",
    "morning mist",
    "dry and hot",
]

TIME = [
    "dawn",
    "early morning",
    "midday",
    "afternoon",
    "late afternoon",
    "dusk",
]

SCENE_DETAIL = [
    "drilling residue visible",
    "fresh blast fragments",
    "machinery in background",
    "open-pit bench face behind",
    "tire tracks on haul road",
    "excavator bucket visible",
    "blasting dust settling",
    "broken rock faces",
]

TEXTURE = [
    "detailed surface texture",
    "rough fractured surfaces",
    "sharp angular edges",
    "visible mineral veins",
    "coarse grain texture",
]

NEGATIVE_PROMPT = (
    "cartoon, smooth, water-worn, cobblestone, beach, "
    "blurry, low quality, unrealistic, painting, sketch"
)


def random_prompt(seed: int | None = None, salt: str | None = None) -> str:
    """生成一条随机化的推理 prompt。"""
    effective_seed = _stable_seed(seed=seed, salt=salt)
    rng = random.Random(effective_seed)
    return (
        f"{TRIGGER_WORD}, open-pit mine muck pile after blasting, "
        f"fresh angular rock fragments, "
        f"{rng.choice(LIGHTING)}, {rng.choice(WEATHER)}, "
        f"{rng.choice(TIME)}, {rng.choice(SCENE_DETAIL)}, "
        f"{rng.choice(TEXTURE)}, photorealistic"
    )


def get_negative_prompt() -> str:
    return NEGATIVE_PROMPT


def _stable_seed(seed: int | None, salt: str | None) -> int | None:
    """Derive a reproducible RNG seed independent of Python's hash randomization."""
    if seed is None and salt is None:
        return None
    payload = f"{salt or ''}:{seed if seed is not None else ''}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
