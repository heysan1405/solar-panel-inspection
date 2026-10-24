# PVEL-AD class mapping investigation

## Conclusion

The original PVEL-AD class list defines 12 zero-based classes. The current
`yolo_dataset\data.yaml` defines eight classes, retaining original classes
0–4 and 8, 9, and 11, with the retained classes renumbered contiguously.
The 138 validation label files contain annotations written using original
class IDs 8–11 rather than the corresponding IDs in the eight-class YAML.

All 260 affected validation annotation rows were compared with their
same-stem original Pascal VOC XML annotation. In every case, the YOLO box
coordinates matched a source XML box after normalization, and its source XML
object name confirmed the class mapping below. Thus these class names are
verified from both the original ordered class list and the corresponding
object annotations.

## Mapping table

| Original class ID | Verified original class name | ID/name in `yolo_dataset\data.yaml` | Retained among the eight? | Validation rows checked |
|---:|---|---|---|---:|
| 8 | `horizontal_dislocation` | 5 / `horizontal_dislocation` | Yes; remap 8 → 5 | 120; all 120 boxes matched source XML |
| 9 | `vertical_dislocation` | 6 / `vertical_dislocation` | Yes; remap 9 → 6 | 36; all 36 boxes matched source XML |
| 10 | `printing_error` | No corresponding class | No | 16; all 16 boxes matched source XML |
| 11 | `short_circuit` | 7 / `short_circuit` | Yes; remap 11 → 7 | 88; all 88 boxes matched source XML |

The row totals cover 138 label files. `printing_error` is not among the eight
configured classes and must not be relabeled as another class; resolving those
annotations requires deciding how to handle a class excluded from this
dataset configuration.

## Original list compared with the current YAML

The original [annotation_classes.txt](./solar_cell_EL_image/PVELAD/EL2021/annotation_classes.txt)
lists, in order:

| Original ID | Original class | Current YAML ID | Current YAML class |
|---:|---|---:|---|
| 0 | `crack` | 0 | `crack` |
| 1 | `finger` | 1 | `finger` |
| 2 | `black_core` | 2 | `black_core` |
| 3 | `thick_line` | 3 | `thick_line` |
| 4 | `star_crack` | 4 | `star_crack` |
| 5 | `corner` | — | Not retained |
| 6 | `fragment` | — | Not retained |
| 7 | `scratch` | — | Not retained |
| 8 | `horizontal_dislocation` | 5 | Retained and renumbered |
| 9 | `vertical_dislocation` | 6 | Retained and renumbered |
| 10 | `printing_error` | — | Not retained |
| 11 | `short_circuit` | 7 | Retained and renumbered |

The current [data.yaml](./yolo_dataset/data.yaml) confirms the eight configured
names and IDs. The original project [README](./PVEL-AD-main/PVEL-AD-main/README.md)
also describes PVEL-AD as having 12 defect categories and notes that
horizontal and vertical dislocation box annotations were added to the dataset.

## Evidence and method

- Read the original ordered `annotation_classes.txt`, the current dataset YAML,
  and the original project README.
- Enumerated object names in the 4,500 XML files under
  `solar_cell_EL_image\PVELAD\EL2021\trainval\Annotations`. The source objects
  include `horizontal_dislocation`, `vertical_dislocation`, `printing_error`,
  and `short_circuit`.
- Matched the 900 validation label files by stem to the original XML
  annotations. Each of the 138 affected labels had a same-stem XML file.
- For every validation label row with ID 8, 9, 10, or 11, compared its
  normalized bounding box with the source XML boxes and checked the matching
  XML object's name. All 260 rows matched a source box (within 2×10⁻⁶ per
  normalized coordinate):
  - ID 8 → `horizontal_dislocation`: 120 rows, across 40 files.
  - ID 9 → `vertical_dislocation`: 36 rows, across 9 files.
  - ID 10 → `printing_error`: 16 rows, in 1 file.
  - ID 11 → `short_circuit`: 88 rows, across 88 files.

This establishes what the extra IDs mean and which have counterparts in the
current class set. It does not change any dataset files or decide how to handle
the excluded `printing_error` class.

## Sources

- [Original annotation class order](./solar_cell_EL_image/PVELAD/EL2021/annotation_classes.txt)
- [Original source annotations](./solar_cell_EL_image/PVELAD/EL2021/trainval/Annotations/)
- [Current eight-class dataset YAML](./yolo_dataset/data.yaml)
- [Original PVEL-AD README](./PVEL-AD-main/PVEL-AD-main/README.md)
