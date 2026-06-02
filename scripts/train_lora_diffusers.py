#!/usr/bin/env python3
"""Lightweight SDXL UNet LoRA trainer using diffusers/PEFT.

This is a fallback path for environments where kohya sd-scripts is not
available. It trains a rank-N attention LoRA on SDXL's UNet and saves a
diffusers-compatible safetensors file that can be loaded by
StableDiffusionXL*Pipeline.load_lora_weights().
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_LORA_CONFIG = PROJECT_ROOT / "configs" / "lora" / "sdxl_rank32.yaml"
DEFAULT_TRAIN_DATA_DIR = PROJECT_ROOT / "data" / "lora_real" / "kohya_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "lora_weights"


@dataclass(frozen=True)
class CachedSample:
    latent: torch.Tensor
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    add_time_ids: torch.Tensor


class KohyaImageCaptionDataset(Dataset):
    def __init__(self, train_data_dir: Path, resolution: int, max_samples: int | None = None):
        self.train_data_dir = train_data_dir
        self.resolution = resolution
        self.items = self._scan(train_data_dir)
        if max_samples is not None:
            self.items = self.items[:max_samples]
        if not self.items:
            raise ValueError(f"No image/caption pairs found under {train_data_dir}")

    def _scan(self, train_data_dir: Path) -> list[tuple[Path, Path]]:
        if not train_data_dir.exists():
            raise FileNotFoundError(train_data_dir)
        pairs: list[tuple[Path, Path]] = []
        for concept_dir in sorted(path for path in train_data_dir.iterdir() if path.is_dir()):
            for image_path in sorted(concept_dir.iterdir(), key=lambda path: path.name.lower()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                caption_path = image_path.with_suffix(".txt")
                if not caption_path.exists():
                    raise FileNotFoundError(f"Missing caption for {image_path}: {caption_path}")
                pairs.append((image_path, caption_path))
        return pairs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, caption_path = self.items[index]
        image = load_square_image(image_path, self.resolution)
        caption = caption_path.read_text(encoding="utf-8").strip()
        return {
            "pixel_values": image,
            "caption": caption,
            "image_path": str(image_path),
        }


class CachedTrainingDataset(Dataset):
    def __init__(self, samples: list[CachedSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CachedSample:
        return self.samples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SDXL UNet LoRA with diffusers.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--lora-config", type=Path, default=DEFAULT_LORA_CONFIG)
    parser.add_argument("--train-data-dir", type=Path, default=DEFAULT_TRAIN_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="mine_blast_pile")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=("fp16", "bf16", "no"), default=None)
    parser.add_argument("--save-every-n-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-batch-size", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def resolve_cache_dir() -> Path | None:
    candidate = PROJECT_ROOT / "weights" / "huggingface" / "hub"
    return candidate if candidate.exists() else None


def load_square_image(path: Path, resolution: int) -> torch.Tensor:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    scale = resolution / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - resolution) // 2)
    top = max(0, (resized.height - resolution) // 2)
    cropped = resized.crop((left, top, left + resolution, top + resolution))

    data = torch.ByteTensor(torch.ByteStorage.from_buffer(cropped.tobytes()))
    data = data.view(resolution, resolution, 3).permute(2, 0, 1).float()
    return data.div(127.5).sub(1.0)


def collate_images(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "captions": [item["caption"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def collate_cached(batch: list[CachedSample]) -> dict[str, torch.Tensor]:
    return {
        "latents": torch.stack([item.latent for item in batch]),
        "prompt_embeds": torch.stack([item.prompt_embeds for item in batch]),
        "pooled_prompt_embeds": torch.stack([item.pooled_prompt_embeds for item in batch]),
        "add_time_ids": torch.stack([item.add_time_ids for item in batch]),
    }


def dtype_from_precision(precision: str, device: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16" and device.startswith("cuda"):
        return torch.float16
    return torch.float32


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def cache_training_samples(
    dataset: KohyaImageCaptionDataset,
    base_model: str,
    vae_model: str,
    cache_dir: Path | None,
    resolution: int,
    device: str,
    weight_dtype: torch.dtype,
    cache_batch_size: int,
    num_workers: int,
) -> list[CachedSample]:
    print(f"Loading SDXL pipeline on CPU for embedding cache: {base_model}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        vae=AutoencoderKL.from_pretrained(
            vae_model,
            torch_dtype=weight_dtype,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=cache_dir is not None,
        ),
        torch_dtype=weight_dtype,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=cache_dir is not None,
    )
    pipe.set_progress_bar_config(disable=True)

    pipe.vae.to(device=device, dtype=weight_dtype)
    pipe.text_encoder.to(device=device, dtype=weight_dtype)
    pipe.text_encoder_2.to(device=device, dtype=weight_dtype)
    pipe.unet.to("cpu")

    loader = DataLoader(
        dataset,
        batch_size=cache_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images,
    )
    vae_scale = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
    add_time_ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        dtype=weight_dtype,
        device="cpu",
    )

    samples: list[CachedSample] = []
    for batch_index, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
        latents = pipe.vae.encode(pixel_values).latent_dist.sample() * vae_scale
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=batch["captions"],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        for item_index in range(latents.shape[0]):
            samples.append(
                CachedSample(
                    latent=latents[item_index].detach().cpu(),
                    prompt_embeds=prompt_embeds[item_index].detach().cpu(),
                    pooled_prompt_embeds=pooled_prompt_embeds[item_index].detach().cpu(),
                    add_time_ids=add_time_ids.clone(),
                )
            )
        print(f"Cached {len(samples)}/{len(dataset)} samples", flush=True)

    del pipe
    free_memory()
    return samples


def save_lora(unet: UNet2DConditionModel, output_dir: Path, output_name: str, step: int | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_name = f"{output_name}.safetensors" if step is None else f"{output_name}_step{step}.safetensors"
    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=lora_state_dict,
        weight_name=weight_name,
        safe_serialization=True,
    )
    return output_dir / weight_name


def write_training_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    base_cfg = load_yaml(args.base_config)
    lora_cfg = load_yaml(args.lora_config)
    base_model = str(base_cfg["sdxl"]["base_model"])
    vae_model = str(base_cfg["sdxl"]["vae"])
    cache_dir = resolve_cache_dir()

    resolution = int(args.resolution or lora_cfg.get("resolution", 1024))
    max_train_steps = int(args.max_train_steps or lora_cfg.get("total_steps", 2500))
    batch_size = int(args.batch_size or lora_cfg.get("batch_size", 1))
    gradient_accumulation = int(args.gradient_accumulation or lora_cfg.get("gradient_accumulation", 4))
    learning_rate = float(args.learning_rate or lora_cfg.get("unet_lr", 1e-4))
    rank = int(args.rank or lora_cfg.get("rank", 32))
    alpha = int(args.alpha or lora_cfg.get("alpha", rank))
    mixed_precision = args.mixed_precision or lora_cfg.get("mixed_precision", "fp16")
    weight_dtype = dtype_from_precision(mixed_precision, args.device)

    dataset = KohyaImageCaptionDataset(
        resolve_project_path(args.train_data_dir),
        resolution=resolution,
        max_samples=args.max_samples,
    )
    print("== Diffusers SDXL LoRA training ==")
    print(f"Samples       : {len(dataset)}")
    print(f"Resolution    : {resolution}")
    print(f"Steps         : {max_train_steps}")
    print(f"Batch size    : {batch_size}")
    print(f"Grad accum    : {gradient_accumulation}")
    print(f"Rank/alpha    : {rank}/{alpha}")
    print(f"Learning rate : {learning_rate}")
    print(f"Device/dtype  : {args.device}/{weight_dtype}")
    print(f"HF cache      : {cache_dir}")

    cached_samples = cache_training_samples(
        dataset=dataset,
        base_model=base_model,
        vae_model=vae_model,
        cache_dir=cache_dir,
        resolution=resolution,
        device=args.device,
        weight_dtype=weight_dtype,
        cache_batch_size=args.cache_batch_size,
        num_workers=args.num_workers,
    )

    print("Loading UNet for LoRA training...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        base_model,
        subfolder="scheduler",
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=cache_dir is not None,
    )
    unet = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=cache_dir is not None,
    )
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet.to(device=args.device, dtype=weight_dtype)
    unet.train()

    trainable_params = [param for param in unet.parameters() if param.requires_grad]
    for param in trainable_params:
        param.data = param.data.float()
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    cached_dataset = CachedTrainingDataset(cached_samples)
    train_loader = DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_cached,
    )

    epoch = 0
    global_step = 0
    running_loss = 0.0
    started_at = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    while global_step < max_train_steps:
        epoch += 1
        for batch in train_loader:
            latents = batch["latents"].to(args.device, dtype=weight_dtype)
            prompt_embeds = batch["prompt_embeds"].to(args.device, dtype=weight_dtype)
            pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(args.device, dtype=weight_dtype)
            add_time_ids = batch["add_time_ids"].to(args.device, dtype=weight_dtype)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                int(noise_scheduler.config.num_train_timesteps),
                (latents.shape[0],),
                device=args.device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                },
            ).sample
            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            (loss / gradient_accumulation).backward()
            running_loss += float(loss.detach().cpu())

            if (global_step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 10 == 0 or global_step == 1:
                elapsed = time.perf_counter() - started_at
                mean_loss = running_loss / max(1, min(global_step, 10))
                running_loss = 0.0
                print(
                    f"step={global_step}/{max_train_steps} "
                    f"epoch={epoch} loss={mean_loss:.6f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if args.save_every_n_steps and global_step % args.save_every_n_steps == 0:
                saved = save_lora(unet, resolve_project_path(args.output_dir), args.output_name, step=global_step)
                print(f"Saved checkpoint: {saved}", flush=True)
            if global_step >= max_train_steps:
                break

    final_path = save_lora(unet, resolve_project_path(args.output_dir), args.output_name)
    elapsed = time.perf_counter() - started_at
    metadata = {
        "base_model": base_model,
        "vae": vae_model,
        "train_data_dir": str(resolve_project_path(args.train_data_dir)),
        "output_path": str(final_path),
        "samples": len(dataset),
        "resolution": resolution,
        "steps": max_train_steps,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "rank": rank,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "mixed_precision": mixed_precision,
        "elapsed_seconds": round(elapsed, 3),
    }
    write_training_metadata(final_path.with_suffix(".json"), metadata)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Saved LoRA: {final_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
