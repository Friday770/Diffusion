# Dataset Delivery Report

This template is for the final large-scale delivery after real generation, QC, COCO assembly, and statistics have completed. No large-scale run results have been recorded here yet.

## Run Inputs

- Unity source directory:
- Inference config:
- QC threshold config:
- Seeds per sample:
- Run start time:
- Run end time:

## Pipeline Commands

```bash
python scripts/04_run_inference.py --config configs/inference/2cn_depth_canny.yaml --seeds 4
python scripts/05_run_qc.py --qc-config configs/qc/thresholds.yaml
python scripts/06_build_coco_dataset.py
python scripts/dataset_statistics.py \
  --annotations data/final_dataset/train/annotations.json \
  --image-dir data/final_dataset/train/images \
  --output-dir docs/dataset_samples
```

## Delivery Summary

- Final image count:
- Final annotation count:
- Average instances per image:
- QC pass rate:
- Dataset disk usage:
- COCO annotations path:
- Statistics report path:
- Spot-check overlay directory:

## Validation Checklist

- [ ] `annotations.json` loads with `pycocotools.coco.COCO`.
- [ ] `image_id` values are globally unique.
- [ ] `annotation_id` values are globally unique.
- [ ] Number of images equals the count of `qc.json` files with `passed=true`.
- [ ] Random spot-check overlays show masks aligned with generated images.
- [ ] Area distribution, per-image instance counts, and resolution distribution are reviewed.
- [ ] Stage-four segmentation loader can read this dataset and start a one-epoch training run.

## Notes

- Do not fill this report with estimated values. Record only values produced by the final run logs and statistics script.
