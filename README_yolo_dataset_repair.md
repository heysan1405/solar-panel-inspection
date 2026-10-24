# Safe YOLO dataset repair

This repair creates `yolo_dataset_repaired_8class` as a separate copy. It
does not modify or remove files in the original `yolo_dataset`.

## Reproduce

Run these commands from the project directory in PowerShell:

```powershell
python .\repair_yolo_dataset.py --plan-only
python .\repair_yolo_dataset.py
```

The first command audits both splits and prints the proposed row counts without
copying or creating files. The second creates the repaired copy, reports, and
separate YAML, then checks every image/label pair using Ultralytics'
`verify_image_label` directly (no cache read or creation). It fails before
copying if class IDs, label syntax/coordinates, split pairing, or source XML
class/box evidence is unexpected. The available output directory must not
already exist unless it is the exact prior result with a matching manifest.

Run the second command again to verify idempotency. If the prior output is
intact and matches its manifest, the script verifies it and makes no changes.
If a staged `.building` directory or a different existing output is found, the
script refuses to overwrite or remove it; inspect it manually.

Dependencies used by the script: Python 3.13, `PyYAML`, `numpy`, and
`ultralytics` (including its installed image decoders).

## Applied class policy

The script checks every annotation against the original same-stem Pascal VOC
XML object class and bounding box before taking action. This check is run
independently on train and validation labels.

- Original IDs 8, 9, and 11 are remapped to current IDs 5, 6, and 7.
- Original IDs 5 (`corner`), 6 (`fragment`), 7 (`scratch`), and 10
  (`printing_error`) are excluded because they are not in the configured
  eight-class task. Every excluded row and original source label path is
  recorded in `excluded_annotations.csv`.
- Retained rows with IDs 0–4 are preserved exactly. For remapped rows, only
  the class-ID token changes; excluded rows are listed in the CSV report.
- Source `*.cache` files are not copied. Images and other unchanged dataset
  files are copied byte-for-byte and verified against their source hashes.

## Outputs

- `yolo_dataset_repaired_8class\` — separate dataset copy.
- `yolo_dataset_repaired_8class\data.yaml` — points to the copy and retains
  the original eight configured names and IDs.
- `yolo_dataset_repaired_8class\repair_manifest.json` — source inventory,
  mapping, and output integrity/verification record used for safe reruns.
- `excluded_annotations.csv` — each excluded source row and its path.
- `yolo_dataset_repair_report.md` — before/after class counts, remap and
  exclusion totals, and uncached Ultralytics verification results.

The script does not train a model or touch model, weight, training, or
experiment files.
