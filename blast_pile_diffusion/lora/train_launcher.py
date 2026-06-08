"""封装 sd-scripts 命令行调用，从 YAML 配置启动 LoRA 训练。"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml


@dataclass(frozen=True)
class TrainingInputCheck:
    train_data_dir: Path
    sd_scripts_dir: Path | None
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.warnings


@dataclass(frozen=True)
class LoraWeightValidation:
    path: Path
    exists: bool
    size_mb: float
    min_mb: float
    max_mb: float

    @property
    def size_ok(self) -> bool:
        return self.min_mb <= self.size_mb <= self.max_mb

    @property
    def valid(self) -> bool:
        return self.exists and self.size_ok


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _as_bool_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def build_train_command(
    config_path: Path,
    base_config_path: Path,
    train_data_dir: Path,
    output_dir: Path,
    output_name: str = "mine_blast_pile",
    pretrained_model_name_or_path: str | Path | None = None,
    tokenizer_cache_dir: Path | None = None,
) -> list[str]:
    """根据配置文件生成 sd-scripts 的训练命令行。"""
    cfg = _load_yaml(config_path)
    base = _load_yaml(base_config_path)

    sdxl_model = pretrained_model_name_or_path or base["sdxl"]["base_model"]
    total_steps = int(cfg["total_steps"])
    warmup_ratio = float(cfg.get("warmup_ratio", 0.05))
    mixed_precision = str(cfg.get("mixed_precision", "fp16"))

    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        mixed_precision,
        "--dynamo_backend",
        "no",
        "sdxl_train_network.py",
        "--pretrained_model_name_or_path",
        str(sdxl_model),
        "--train_data_dir",
        str(train_data_dir),
        "--output_dir",
        str(output_dir),
        "--output_name",
        output_name,
        "--save_model_as",
        str(cfg.get("save_model_as", "safetensors")),
        "--network_module",
        str(cfg.get("network_module", "networks.lora")),
        "--network_dim",
        str(cfg["rank"]),
        "--network_alpha",
        str(cfg["alpha"]),
        "--optimizer_type",
        str(cfg.get("optimizer", "AdamW8bit")),
        "--learning_rate",
        str(cfg["unet_lr"]),
        "--text_encoder_lr",
        str(cfg["text_encoder_lr"]),
        "--lr_scheduler",
        str(cfg.get("lr_scheduler", "cosine")),
        "--lr_warmup_steps",
        str(int(total_steps * warmup_ratio)),
        "--max_train_steps",
        str(total_steps),
        "--train_batch_size",
        str(cfg["batch_size"]),
        "--gradient_accumulation_steps",
        str(cfg.get("gradient_accumulation", 4)),
        "--resolution",
        str(cfg["resolution"]),
        "--caption_extension",
        str(cfg.get("caption_extension", ".txt")),
        "--bucket_reso_steps",
        str(cfg.get("bucket_reso_steps", 64)),
        "--min_bucket_reso",
        str(cfg.get("min_bucket_reso", 768)),
        "--max_bucket_reso",
        str(cfg.get("max_bucket_reso", 1024)),
        "--mixed_precision",
        mixed_precision,
        "--seed",
        str(cfg.get("seed", 42)),
        "--save_every_n_steps",
        str(cfg.get("save_every_n_steps", 500)),
        "--logging_dir",
        str(cfg.get("logging_dir", "logs")),
    ]

    if "max_data_loader_n_workers" in cfg:
        cmd.extend(["--max_data_loader_n_workers", str(cfg["max_data_loader_n_workers"])])

    if tokenizer_cache_dir is not None:
        cmd.extend(["--tokenizer_cache_dir", str(tokenizer_cache_dir)])

    _as_bool_flag(cmd, bool(cfg.get("enable_bucket", True)), "--enable_bucket")
    _as_bool_flag(cmd, bool(cfg.get("cache_latents", True)), "--cache_latents")
    _as_bool_flag(cmd, bool(cfg.get("cache_latents_to_disk", True)), "--cache_latents_to_disk")
    _as_bool_flag(cmd, bool(cfg.get("shuffle_caption", True)), "--shuffle_caption")
    _as_bool_flag(cmd, bool(cfg.get("gradient_checkpointing", False)), "--gradient_checkpointing")
    _as_bool_flag(cmd, bool(cfg.get("sdpa", False)), "--sdpa")
    _as_bool_flag(cmd, bool(cfg.get("xformers", False)), "--xformers")
    _as_bool_flag(
        cmd,
        bool(cfg.get("persistent_data_loader_workers", False)),
        "--persistent_data_loader_workers",
    )

    return cmd


def check_training_inputs(
    train_data_dir: Path,
    sd_scripts_dir: Path | None = None,
) -> TrainingInputCheck:
    """Collect non-fatal warnings for dry-run and preflight output."""
    warnings: list[str] = []

    if not train_data_dir.exists():
        warnings.append(f"Training data directory does not exist: {train_data_dir}")
    elif not any(train_data_dir.glob("*_*/*")):
        warnings.append(f"Training data directory has no kohya concept files: {train_data_dir}")

    if sd_scripts_dir is not None:
        train_script = sd_scripts_dir / "sdxl_train_network.py"
        if not sd_scripts_dir.exists():
            warnings.append(f"sd-scripts directory does not exist: {sd_scripts_dir}")
        elif not train_script.exists():
            warnings.append(f"sdxl_train_network.py not found under: {sd_scripts_dir}")

    return TrainingInputCheck(
        train_data_dir=train_data_dir,
        sd_scripts_dir=sd_scripts_dir,
        warnings=warnings,
    )


def validate_lora_weight(
    lora_path: Path,
    min_mb: float = 50.0,
    max_mb: float = 250.0,
) -> LoraWeightValidation:
    """Validate that a trained SDXL LoRA weight file has a plausible size."""
    exists = lora_path.exists()
    size_mb = (lora_path.stat().st_size / (1024 * 1024)) if exists else 0.0
    return LoraWeightValidation(
        path=lora_path,
        exists=exists,
        size_mb=round(size_mb, 2),
        min_mb=min_mb,
        max_mb=max_mb,
    )


def format_command(cmd: Sequence[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def launch_training(
    config_path: Path,
    base_config_path: Path,
    train_data_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    sd_scripts_dir: Path | None = None,
    output_name: str = "mine_blast_pile",
    pretrained_model_name_or_path: str | Path | None = None,
    tokenizer_cache_dir: Path | None = None,
) -> int:
    """启动 LoRA 训练，返回 exit code。

    sd_scripts_dir 应指向克隆的 kohya-ss/sd-scripts 目录。
    """
    cmd = build_train_command(
        config_path=config_path,
        base_config_path=base_config_path,
        train_data_dir=train_data_dir,
        output_dir=output_dir,
        output_name=output_name,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        tokenizer_cache_dir=tokenizer_cache_dir,
    )
    check = check_training_inputs(train_data_dir=train_data_dir, sd_scripts_dir=sd_scripts_dir)

    print("==========================================")
    print(" LoRA training command")
    print("==========================================")
    print(format_command(cmd))
    print("==========================================")

    if check.warnings:
        print("Preflight warnings:")
        for warning in check.warnings:
            print(f"  - {warning}")

    if dry_run:
        print("[dry-run] command built; no training was started")
        return 0

    if check.warnings:
        raise RuntimeError("Training inputs are incomplete; run with --dry-run to inspect command.")

    output_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd,
        check=False,
        cwd=str(sd_scripts_dir) if sd_scripts_dir is not None else None,
    )
    return result.returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SDXL LoRA training via sd-scripts.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora/sdxl_5090_32gb.yaml"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--train-data-dir", type=Path, default=Path("data/lora_real/kohya_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("lora_weights"))
    parser.add_argument("--output-name", default="mine_blast_pile")
    parser.add_argument("--sd-scripts-dir", type=Path, default=None)
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=Path,
        default=None,
        help="Optional local SDXL diffusers directory or model id.",
    )
    parser.add_argument(
        "--tokenizer-cache-dir",
        type=Path,
        default=None,
        help="Optional sd-scripts tokenizer cache directory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sd_scripts_dir = args.sd_scripts_dir
    if sd_scripts_dir is None and os.environ.get("SD_SCRIPTS_DIR"):
        sd_scripts_dir = Path(os.environ["SD_SCRIPTS_DIR"])

    exit_code = launch_training(
        config_path=args.config,
        base_config_path=args.base_config,
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        sd_scripts_dir=sd_scripts_dir,
        output_name=args.output_name,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
    )

    if args.validate_output and exit_code == 0 and not args.dry_run:
        validation = validate_lora_weight(args.output_dir / f"{args.output_name}.safetensors")
        print(
            "LoRA weight validation: "
            f"exists={validation.exists}, size_mb={validation.size_mb}, "
            f"expected={validation.min_mb}-{validation.max_mb}MB"
        )
        if not validation.valid:
            return 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
