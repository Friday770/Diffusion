from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from blast_pile_diffusion.inference import pipeline_builder


def _inference_cfg(lora_path: Path | None = None) -> dict:
    cfg = {
        "controlnets": [
            {"name": "depth", "model": "mock-depth", "scale": 0.7},
            {"name": "canny", "model": "mock-canny", "scale": 0.6},
        ],
        "inference": {
            "strength": 0.45,
            "num_inference_steps": 4,
            "guidance_scale": 6.5,
            "memory": {"enable_model_cpu_offload": False},
        },
    }
    if lora_path is not None:
        cfg["lora"] = {"path": str(lora_path), "weight": 0.8}
    return cfg


def _base_cfg() -> dict:
    return {
        "dtype": "float16",
        "sdxl": {
            "base_model": "mock-sdxl",
            "vae": "mock-vae",
        },
    }


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


class FakeControlNetModel:
    loaded: list[tuple[str, torch.dtype]] = []

    @classmethod
    def from_pretrained(cls, model: str, torch_dtype: torch.dtype):
        cls.loaded.append((model, torch_dtype))
        return {"model": model, "dtype": torch_dtype}


class FakeAutoencoderKL:
    loaded: list[tuple[str, torch.dtype]] = []

    @classmethod
    def from_pretrained(cls, model: str, torch_dtype: torch.dtype):
        cls.loaded.append((model, torch_dtype))
        return {"vae": model, "dtype": torch_dtype}


class FakePipeline:
    last_instance = None

    def __init__(self, base_model: str, controlnet: list, vae: dict, torch_dtype: torch.dtype):
        self.base_model = base_model
        self.controlnet = controlnet
        self.vae = vae
        self.torch_dtype = torch_dtype
        self.device = None
        self.lora_loaded = None
        self.adapters = None
        self.cpu_offload = False
        FakePipeline.last_instance = self

    @classmethod
    def from_pretrained(cls, base_model: str, controlnet: list, vae: dict, torch_dtype: torch.dtype):
        return cls(base_model, controlnet, vae, torch_dtype)

    def to(self, device: str):
        self.device = device
        return self

    def enable_model_cpu_offload(self) -> None:
        self.cpu_offload = True

    def load_lora_weights(self, path: str, adapter_name: str) -> None:
        self.lora_loaded = {"path": path, "adapter_name": adapter_name}

    def set_adapters(self, adapters: list[str], adapter_weights: list[float]) -> None:
        self.adapters = {"adapters": adapters, "adapter_weights": adapter_weights}


@pytest.fixture(autouse=True)
def fake_diffusers(monkeypatch):
    FakeControlNetModel.loaded = []
    FakeAutoencoderKL.loaded = []
    FakePipeline.last_instance = None
    monkeypatch.setattr(pipeline_builder, "ControlNetModel", FakeControlNetModel)
    monkeypatch.setattr(pipeline_builder, "AutoencoderKL", FakeAutoencoderKL)
    monkeypatch.setattr(pipeline_builder, "StableDiffusionXLControlNetImg2ImgPipeline", FakePipeline)


def test_build_pipeline_filters_controlnets_and_loads_lora(tmp_path: Path) -> None:
    lora_path = tmp_path / "mine.safetensors"
    lora_path.write_bytes(b"mock")
    inference_path = tmp_path / "inference.yaml"
    base_path = tmp_path / "base.yaml"
    _write_yaml(inference_path, _inference_cfg(lora_path=lora_path))
    _write_yaml(base_path, _base_cfg())

    pipe, cfg = pipeline_builder.build_pipeline(
        inference_config_path=inference_path,
        base_config_path=base_path,
        device="cpu",
        controlnet_names=("depth",),
    )

    assert cfg["controlnets"] == [{"name": "depth", "model": "mock-depth", "scale": 0.7}]
    assert FakeControlNetModel.loaded == [("mock-depth", torch.float16)]
    assert FakeAutoencoderKL.loaded == [("mock-vae", torch.float16)]
    assert pipe.base_model == "mock-sdxl"
    assert pipe.device == "cpu"
    assert pipe.lora_loaded == {"path": str(lora_path), "adapter_name": "mine_style"}
    assert pipe.adapters == {"adapters": ["mine_style"], "adapter_weights": [0.8]}


def test_build_pipeline_skip_lora(tmp_path: Path) -> None:
    lora_path = tmp_path / "mine.safetensors"
    lora_path.write_bytes(b"mock")
    inference_path = tmp_path / "inference.yaml"
    base_path = tmp_path / "base.yaml"
    _write_yaml(inference_path, _inference_cfg(lora_path=lora_path))
    _write_yaml(base_path, _base_cfg())

    pipe, _cfg = pipeline_builder.build_pipeline(
        inference_config_path=inference_path,
        base_config_path=base_path,
        device="cpu",
        skip_lora=True,
    )

    assert pipe.lora_loaded is None
    assert len(FakeControlNetModel.loaded) == 2


def test_build_pipeline_required_lora_missing_fails_before_model_load(tmp_path: Path) -> None:
    missing_lora_path = tmp_path / "missing.safetensors"
    inference_cfg = _inference_cfg(lora_path=missing_lora_path)
    inference_cfg["lora"]["required"] = True
    inference_path = tmp_path / "inference.yaml"
    base_path = tmp_path / "base.yaml"
    _write_yaml(inference_path, inference_cfg)
    _write_yaml(base_path, _base_cfg())

    with pytest.raises(FileNotFoundError, match="LoRA"):
        pipeline_builder.build_pipeline(
            inference_config_path=inference_path,
            base_config_path=base_path,
            device="cpu",
        )

    assert FakeControlNetModel.loaded == []
    assert FakeAutoencoderKL.loaded == []


def test_unknown_controlnet_selection_raises(tmp_path: Path) -> None:
    inference_path = tmp_path / "inference.yaml"
    base_path = tmp_path / "base.yaml"
    _write_yaml(inference_path, _inference_cfg())
    _write_yaml(base_path, _base_cfg())

    with pytest.raises(ValueError, match="不存在 ControlNet"):
        pipeline_builder.build_pipeline(
            inference_config_path=inference_path,
            base_config_path=base_path,
            controlnet_names=("normal",),
        )


def test_validate_config_rejects_unknown_controlnet() -> None:
    cfg = _inference_cfg()
    cfg["controlnets"][0]["name"] = "rgb"

    with pytest.raises(ValueError, match="未知 ControlNet"):
        pipeline_builder.validate_inference_config(cfg)
