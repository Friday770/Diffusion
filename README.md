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
lora_weights/mine_blast_pile.safetensors
```

should be downloaded or copied into that directory separately.

## Weight Files

Model weights are managed separately from this code repository. The LoRA weights
will be stored in a separate GitHub repository or release artifact created
specifically for weight files.

After obtaining the LoRA weight file, place it at:

```text
lora_weights/mine_blast_pile.safetensors
```

The inference configs expect that path by default.

## LoRA Data Preparation

The expected image directory for LoRA training is:

```text
data/lora_real/images/
```

Prepare captions and the kohya/sd-scripts dataset structure:

```bash
python scripts/02_prepare_lora_data.py \
  --image-dir data/lora_real/images \
  --caption-dir data/lora_real/captions \
  --output-dir data/lora_real/kohya_dataset \
  --clean
```

## LoRA Training

Training is designed to run on a server with `sd-scripts` available:

```bash
python -m blast_pile_diffusion.lora.train_launcher \
  --sd-scripts-dir sd-scripts \
  --train-data-dir data/lora_real/kohya_dataset \
  --output-dir lora_weights \
  --validate-output
```

For 24GB GPUs, a lower-memory SDXL LoRA setting such as rank 16 and 768
resolution is recommended.

## Inference

After placing the LoRA file at `lora_weights/mine_blast_pile.safetensors`, run
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
