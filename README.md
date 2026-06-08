# Diffusion

This repository contains the code, configuration files, training helpers, inference
pipeline, quality-control utilities, and tests for the blast-pile diffusion
workflow.

## What Is Included

- `blast_pile_diffusion/`: core Python package
- `configs/`: LoRA, inference, and QC configuration files
- `scripts/`: preprocessing, LoRA data preparation, training, inference, QC, and dataset export scripts
- `tests/`: unit tests
- `docs/`: project documentation and architecture figures
- `lora_weights/.gitkeep`: placeholder for the LoRA weight directory

## What Is Not Included

Large data files and model weights are intentionally not committed to this code
repository.

Ignored paths include:

- `data/`
- `weights/`
- `lora_weights/*.safetensors`
- `*.safetensors`
- `*.ckpt`
- `*.pt`
- `*.pth`
- local environments and caches such as `.venv/`, `.uv-cache/`, `.pytest_cache/`, and `.ruff_cache/`

The LoRA weight directory is kept in the repository as an empty placeholder:

```text
lora_weights/
  .gitkeep
```

The actual LoRA weights, for example:

```text
lora_weights/mine_blast_pile_v3_diffusers.safetensors
```

should be downloaded or copied into that directory separately.

## Weight Files

Model weights are managed separately from this code repository. The LoRA weights
will be stored in a separate GitHub repository or release artifact created
specifically for weight files.

After obtaining the LoRA weight file, place it at:

```text
lora_weights/mine_blast_pile_v3_diffusers.safetensors
```

The inference configs expect that path by default.

## LoRA Data Preparation

The current LoRA source dataset is expected at:

```text
PhotoForLoRA/
```

The folder should contain paired image and prompt files:

```text
PhotoForLoRA/
  0001.jpg
  0001.txt
  0002.jpg
  0002.txt
```

Prepare the kohya/sd-scripts dataset structure without overwriting the existing
prompts:

```bash
python scripts/02_prepare_lora_data.py \
  --image-dir PhotoForLoRA \
  --caption-dir PhotoForLoRA \
  --output-dir data/lora_real/kohya_dataset \
  --clean
```

## LoRA Training

Training is designed to run on a server with `sd-scripts` available:

```bash
python -m blast_pile_diffusion.lora.train_launcher \
  --config configs/lora/sdxl_5090_32gb.yaml \
  --sd-scripts-dir sd-scripts \
  --train-data-dir data/lora_real/kohya_dataset \
  --output-dir lora_weights \
  --pretrained-model-name-or-path weight/sdxl_base.safetensors \
  --validate-output
```

The default training config is tuned for a 32GB RTX 5090-class GPU. It uses SDXL
at 1024 resolution, rank 32, bf16 mixed precision, SDPA attention, cached
latents, and an effective batch size of 4 (`batch_size=2`,
`gradient_accumulation=2`). If the installed `sd-scripts` version does not
support `--sdpa`, set `sdpa: false` in `configs/lora/sdxl_5090_32gb.yaml`.

For 24GB GPUs, use the more conservative `configs/lora/sdxl_rank32.yaml` profile
or reduce batch size, resolution, or rank.

## Inference

After placing the LoRA file at `lora_weights/mine_blast_pile_v3_diffusers.safetensors`, run
inference with one of the provided configs:

```bash
python scripts/04_run_inference.py \
  --config configs/inference/2cn_depth_canny.yaml \
  --input-dir data/preprocessed \
  --output-dir data/generated
```

## Tests

Run the test suite with:

```bash
pytest
```
