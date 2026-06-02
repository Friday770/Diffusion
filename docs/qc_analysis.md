# QC Analysis Template

> This document is a template for E3 reporting. Fill it only after running
> `scripts/05_run_qc.py` on real generated data. Do not treat the placeholders
> below as experimental conclusions.

## Run Metadata

- QC report: `data/generated/qc_report.json`
- QC config: `configs/qc/thresholds.yaml`
- Generated data range:
- Run date:
- Reviewer:

## Overall Summary

- Total generated samples:
- Passed:
- Failed:
- Errors:
- Pass rate:
- Mean / median / p95 / p99 edge offset:

## Per-Scene Distribution

List scenes whose pass rate is much lower than the dataset average. Any scene
below `reporting.low_scene_pass_rate_pct` should be inspected first.

| Scene ID | Total | Passed | Failed | Errors | Pass Rate | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TODO |  |  |  |  |  |  |

## Failure Review

Review failed samples with `debug_overlay.png`.

| Failure Category | Count | Evidence / Notes |
| --- | ---: | --- |
| Edge misalignment |  |  |
| Blurry generation |  |  |
| Severe artifacts |  |  |
| Missing/corrupt inputs |  |  |

## Borderline Samples

Record manual checks for samples near the pass/fail threshold from E2.

| Sample | Mean Offset | P99 Offset | QC Decision | Manual Decision | Notes |
| --- | ---: | ---: | --- | --- | --- |
| TODO |  |  |  |  |  |

## Follow-Up Actions

- Threshold changes needed:
- Scene-specific issues:
- Inference parameter changes needed:
- Data regeneration needed:
