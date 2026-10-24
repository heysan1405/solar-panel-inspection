# Validation dataset audit

This audit is limited to the YOLO validation split. It does not change images,
labels, split membership, model files, or training code.

## Reproduce the audit

From the project directory, run:

```powershell
python .\audit_validation_dataset.py --data-yaml .\yolo_dataset\data.yaml
```

The script checks every validation image with Pillow and OpenCV, checks its
corresponding YOLO detection label, finds orphan labels, and checks whether
`labels\val.cache` agrees with the current image/label paths and findings. It
writes:

- `validation_audit.csv` — per-file findings, paths, details, and repairability
- `validation_audit_summary.txt` — totals and the cached Ultralytics messages

The audit requires Python packages `Pillow`, `opencv-python`, `numpy`, `PyYAML`,
and `ultralytics`.

## Findings and disposition

The initial audit inspected 900 validation images. All 900 images are readable
by both image decoders; all corresponding label files exist. It found 138
invalid label files containing 260 annotation rows with class IDs 8, 9, 10, or
11, outside the configured IDs 0–7. The YAML defines eight classes and gives no
mapping for those additional IDs, so changing the class IDs would invent
annotation semantics. The 260 affected rows are ID 8 (120 rows), ID 9 (36),
ID 10 (16), and ID 11 (88). Those files are left unchanged for manual review.

No image or label repair was safe to make. Therefore there is no repaired
dataset directory or alternate dataset YAML. The original
`yolo_dataset\data.yaml` remains the only dataset YAML; it will continue to
produce the 138 label-corruption warnings until the class mapping is verified
and the annotations are corrected using authoritative source information.

The existing `yolo_dataset\labels\val.cache` hash and cached scan results agree
with this audit, so the cache is not stale and was not removed or modified.
After an authorized label correction, rerun the audit; Ultralytics will rebuild
its relevant validation cache if the label paths or file sizes change.
