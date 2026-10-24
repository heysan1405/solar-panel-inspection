# YOLO dataset repair report

- Source dataset: `C:\Users\PEXUS\Downloads\dsa\D8 PVEL AD\D8 PVEL AD\yolo_dataset`
- Repaired dataset: `C:\Users\PEXUS\Downloads\dsa\D8 PVEL AD\D8 PVEL AD\yolo_dataset_repaired_8class`
- Repair scope: train and validation labels; original dataset is unchanged.
- Method: IDs 8, 9, and 11 are remapped only after per-row source XML class-name and bounding-box verification.
- Excluded source classes: IDs 5 (`corner`), 6 (`fragment`), 7 (`scratch`), and 10 (`printing_error`).

## Before/after annotation counts

| Split | Original ID/name | Before | Repaired ID/name | After | Action |
|---|---|---:|---|---:|---|
| train | 0 / `crack` | 1010 | 0 / `crack` | 1010 | preserved |
| train | 1 / `finger` | 2317 | 1 / `finger` | 2317 | preserved |
| train | 2 / `black_core` | 826 | 2 / `black_core` | 826 | preserved |
| train | 3 / `thick_line` | 776 | 3 / `thick_line` | 776 | preserved |
| train | 4 / `star_crack` | 110 | 4 / `star_crack` | 110 | preserved |
| train | 5 / `corner` | 7 | 5 / `—` | 0 | excluded |
| train | 6 / `fragment` | 7 | 6 / `—` | 0 | excluded |
| train | 7 / `scratch` | 2 | 7 / `—` | 0 | excluded |
| train | 8 / `horizontal_dislocation` | 678 | 5 / `horizontal_dislocation` | 678 | remapped 8 → 5 |
| train | 9 / `vertical_dislocation` | 101 | 6 / `vertical_dislocation` | 101 | remapped 9 → 6 |
| train | 10 / `printing_error` | 16 | — / `—` | 0 | excluded |
| train | 11 / `short_circuit` | 404 | 7 / `short_circuit` | 404 | remapped 11 → 7 |
| val | 0 / `crack` | 250 | 0 / `crack` | 250 | preserved |
| val | 1 / `finger` | 641 | 1 / `finger` | 641 | preserved |
| val | 2 / `black_core` | 202 | 2 / `black_core` | 202 | preserved |
| val | 3 / `thick_line` | 205 | 3 / `thick_line` | 205 | preserved |
| val | 4 / `star_crack` | 25 | 4 / `star_crack` | 25 | preserved |
| val | 5 / `corner` | 2 | 5 / `—` | 0 | excluded |
| val | 6 / `fragment` | 0 | 6 / `—` | 0 | excluded |
| val | 7 / `scratch` | 3 | 7 / `—` | 0 | excluded |
| val | 8 / `horizontal_dislocation` | 120 | 5 / `horizontal_dislocation` | 120 | remapped 8 → 5 |
| val | 9 / `vertical_dislocation` | 36 | 6 / `vertical_dislocation` | 36 | remapped 9 → 6 |
| val | 10 / `printing_error` | 16 | — / `—` | 0 | excluded |
| val | 11 / `short_circuit` | 88 | 7 / `short_circuit` | 88 | remapped 11 → 7 |

### Combined split totals

| Repaired class ID | Class name | Before (source ID) | After |
|---:|---|---:|---:|
| 0 | `crack` | 1260 (source ID 0) | 1260 |
| 1 | `finger` | 2958 (source ID 1) | 2958 |
| 2 | `black_core` | 1028 (source ID 2) | 1028 |
| 3 | `thick_line` | 981 (source ID 3) | 981 |
| 4 | `star_crack` | 135 (source ID 4) | 135 |
| 5 | `horizontal_dislocation` | 798 (source ID 8) | 798 |
| 6 | `vertical_dislocation` | 137 (source ID 9) | 137 |
| 7 | `short_circuit` | 492 (source ID 11) | 492 |

## Repair totals

- Remapped annotation rows: **1427** (train: 1183, val: 244).
- Excluded annotation rows: **53** (train: 32, val: 21).
- Excluded original ID 5 (`corner`): 9 rows.
- Excluded original ID 6 (`fragment`): 7 rows.
- Excluded original ID 7 (`scratch`): 5 rows.
- Excluded original ID 10 (`printing_error`): 32 rows.

## Uncached Ultralytics verification

| Split | Images | Label files | Annotation rows | Missing | Empty | Corrupt |
|---|---:|---:|---:|---:|---:|---:|
| train | 3600 | 3600 | 6222 | 0 | 6 | 0 |
| val | 900 | 900 | 1567 | 0 | 1 | 0 |

Result: **4500 images verified; 0 corrupt samples.** Verification used Ultralytics `verify_image_label` directly for each image/label pair, without reading or creating a dataset cache.

## Excluded cache files and unresolved issues

Generated source cache files were not copied: `labels/train.cache`, `labels/val.cache`.
Unresolved class IDs, malformed rows, missing image/label pairs, unreadable source annotations, and unverified source boxes: **none**. All excluded rows are listed in `excluded_annotations.csv`.
