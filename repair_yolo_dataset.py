from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ultralytics.data.utils import verify_image_label


SCRIPT_VERSION = 1
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_REMAP = {8: 5, 9: 6, 11: 7}
EXCLUDED_IDS = {5, 6, 7, 10}
EXPECTED_ORIGINAL_CLASS_COUNT = 12
BOX_TOLERANCE = 2e-6


@dataclass
class RowAction:
    line_number: int
    class_id: int
    class_name: str
    row_text: str
    action: str
    output_class_id: int | None


@dataclass
class SplitAudit:
    name: str
    images: list[Path]
    labels: list[Path]
    before_counts: Counter[int]
    after_counts: Counter[int]
    actions: dict[Path, list[RowAction]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_fingerprint(root: Path) -> tuple[str, list[dict[str, Any]]]:
    entries = []
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p).casefold()):
        stat = path.stat()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = json.dumps(entries, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), entries


def load_source_configuration(
    source: Path, classes_file: Path
) -> tuple[dict[str, Any], list[str], dict[int, str]]:
    config_path = source / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {config_path}")

    class_count = int(config["nc"])
    names_value = config.get("names")
    if isinstance(names_value, list):
        names = {index: str(name) for index, name in enumerate(names_value)}
    elif isinstance(names_value, dict):
        names = {int(index): str(name) for index, name in names_value.items()}
    else:
        raise ValueError(f"Unsupported 'names' value in {config_path}")
    if class_count != 8 or set(names) != set(range(class_count)):
        raise ValueError(
            f"Expected exactly eight configured class IDs 0-7 in {config_path}; "
            f"found nc={class_count}, IDs={sorted(names)}"
        )

    original_names = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(original_names) != EXPECTED_ORIGINAL_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ORIGINAL_CLASS_COUNT} original classes in "
            f"{classes_file}; found {len(original_names)}"
        )

    for original_id in range(5):
        if names[original_id] != original_names[original_id]:
            raise ValueError(
                f"Original class {original_id} ({original_names[original_id]}) "
                f"does not match configured class {original_id} ({names[original_id]})"
            )
    for original_id, configured_id in CLASS_REMAP.items():
        if names[configured_id] != original_names[original_id]:
            raise ValueError(
                f"Verified remap {original_id}->{configured_id} does not match YAML: "
                f"{original_names[original_id]} != {names[configured_id]}"
            )
    if original_names[10] != "printing_error":
        raise ValueError(
            f"Expected original ID 10 to be printing_error, found {original_names[10]!r}"
        )
    excluded_names = {original_names[index] for index in EXCLUDED_IDS}
    if excluded_names != {"corner", "fragment", "scratch", "printing_error"}:
        raise ValueError(f"Unexpected non-task class names for IDs 5,6,7,10: {excluded_names}")

    return config, original_names, names


def list_images(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: str(path).casefold(),
    )


def normalized_stems(paths: list[Path], root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        key = path.relative_to(root).with_suffix("").as_posix().casefold()
        if key in result:
            raise ValueError(f"Duplicate case-insensitive relative stem: {path} and {result[key]}")
        result[key] = path
    return result


def parse_source_boxes(xml_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is None:
            raise ValueError("missing XML <size>")
        image_width = float(size.findtext("width", ""))
        image_height = float(size.findtext("height", ""))
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"invalid source image dimensions {image_width}x{image_height}")
        boxes = []
        for obj in root.findall("object"):
            class_name = obj.findtext("name")
            box = obj.find("bndbox")
            if not class_name or box is None:
                raise ValueError("source object is missing its class name or bounding box")
            xmin = float(box.findtext("xmin", ""))
            ymin = float(box.findtext("ymin", ""))
            xmax = float(box.findtext("xmax", ""))
            ymax = float(box.findtext("ymax", ""))
            boxes.append(
                (
                    class_name,
                    (
                        (xmin + xmax) / (2 * image_width),
                        (ymin + ymax) / (2 * image_height),
                        (xmax - xmin) / image_width,
                        (ymax - ymin) / image_height,
                    ),
                )
            )
        return boxes
    except (ET.ParseError, OSError, TypeError) as exc:
        raise ValueError(f"Cannot read source annotation {xml_path}: {exc}") from exc


def replace_class_token(line: str, replacement: int) -> str:
    newline = line[len(line.rstrip("\r\n")) :]
    body = line[: len(line) - len(newline)] if newline else line
    match = re.match(r"^(\ufeff?\s*)\S+", body)
    if match is None:
        raise ValueError(f"Cannot identify class token in row {line!r}")
    return f"{match.group(1)}{replacement}{body[match.end():]}{newline}"


def read_label_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.readlines()


def audit_split(
    split_name: str,
    source: Path,
    annotations: Path,
    original_names: list[str],
) -> SplitAudit:
    image_root = source / "images" / split_name
    label_root = source / "labels" / split_name
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"Missing {split_name} image or label directory")

    images = list_images(image_root)
    labels = sorted(label_root.rglob("*.txt"), key=lambda path: str(path).casefold())
    image_map = normalized_stems(images, image_root)
    label_map = normalized_stems(labels, label_root)
    if not images:
        raise ValueError(f"No supported images found in {image_root}")
    if set(image_map) != set(label_map):
        missing_labels = sorted(set(image_map) - set(label_map))
        orphan_labels = sorted(set(label_map) - set(image_map))
        raise ValueError(
            f"{split_name} image/label pairing mismatch; "
            f"missing labels={missing_labels[:10]}, orphan labels={orphan_labels[:10]}"
        )

    before_counts: Counter[int] = Counter()
    after_counts: Counter[int] = Counter()
    actions: dict[Path, list[RowAction]] = {}
    problems: list[str] = []
    for label_path in label_map.values():
        xml_path = annotations / label_path.relative_to(label_root).with_suffix(".xml")
        if not xml_path.is_file():
            problems.append(f"{label_path}: missing source XML {xml_path}")
            continue
        try:
            source_boxes = parse_source_boxes(xml_path)
            rows = read_label_lines(label_path)
        except (OSError, UnicodeError, ValueError) as exc:
            problems.append(f"{label_path}: {exc}")
            continue

        file_actions: list[RowAction] = []
        for line_number, raw_line in enumerate(rows, start=1):
            line_text = raw_line.rstrip("\r\n")
            fields = line_text.removeprefix("\ufeff").split()
            if not fields:
                continue
            if len(fields) != 5:
                problems.append(
                    f"{label_path}:{line_number}: expected five YOLO fields, found {len(fields)}"
                )
                continue
            try:
                values = [float(field) for field in fields]
            except ValueError:
                problems.append(f"{label_path}:{line_number}: non-numeric YOLO row {line_text!r}")
                continue
            if not all(math.isfinite(value) for value in values):
                problems.append(f"{label_path}:{line_number}: non-finite YOLO value")
                continue
            if not values[0].is_integer():
                problems.append(
                    f"{label_path}:{line_number}: non-integer class ID {values[0]}"
                )
                continue
            class_id = int(values[0])
            if class_id < 0 or class_id >= len(original_names):
                problems.append(
                    f"{label_path}:{line_number}: unexpected class ID {class_id}; "
                    f"original IDs are 0-{len(original_names) - 1}"
                )
                continue
            coordinates = values[1:]
            if any(value < 0 or value > 1 for value in coordinates):
                problems.append(
                    f"{label_path}:{line_number}: coordinates outside normalized [0,1]"
                )
                continue
            x_center, y_center, width, height = coordinates
            if width <= 0 or height <= 0:
                problems.append(f"{label_path}:{line_number}: zero/negative box size")
                continue
            if (
                x_center - width / 2 < 0
                or y_center - height / 2 < 0
                or x_center + width / 2 > 1
                or y_center + height / 2 > 1
            ):
                problems.append(
                    f"{label_path}:{line_number}: box extends outside normalized image bounds"
                )
                continue

            original_name = original_names[class_id]
            source_match = any(
                source_name == original_name
                and max(abs(actual - expected) for actual, expected in zip(coordinates, box))
                <= BOX_TOLERANCE
                for source_name, box in source_boxes
            )
            if not source_match:
                problems.append(
                    f"{label_path}:{line_number}: class {class_id} ({original_name}) "
                    f"and box are not confirmed by {xml_path}"
                )
                continue

            before_counts[class_id] += 1
            if class_id in EXCLUDED_IDS:
                file_actions.append(
                    RowAction(
                        line_number,
                        class_id,
                        original_name,
                        line_text,
                        "excluded",
                        None,
                    )
                )
            elif class_id in CLASS_REMAP:
                output_id = CLASS_REMAP[class_id]
                after_counts[output_id] += 1
                file_actions.append(
                    RowAction(
                        line_number,
                        class_id,
                        original_name,
                        line_text,
                        "remapped",
                        output_id,
                    )
                )
            else:
                after_counts[class_id] += 1
                file_actions.append(
                    RowAction(
                        line_number,
                        class_id,
                        original_name,
                        line_text,
                        "preserved",
                        class_id,
                    )
                )
        actions[label_path] = file_actions

    if problems:
        raise ValueError(
            f"Preflight found {len(problems)} unresolved {split_name} issue(s); "
            "no repaired copy was created:\n" + "\n".join(problems[:40])
        )
    return SplitAudit(split_name, images, labels, before_counts, after_counts, actions)


def apply_split_repair(audit: SplitAudit, source: Path, stage: Path) -> None:
    for source_label in audit.labels:
        destination = stage / source_label.relative_to(source)
        output_lines: list[str] = []
        by_line = {action.line_number: action for action in audit.actions[source_label]}
        lines = read_label_lines(source_label)
        for line_number, line in enumerate(lines, start=1):
            action = by_line.get(line_number)
            if action is None:
                output_lines.append(line)
            elif action.action == "excluded":
                continue
            elif action.action == "remapped":
                output_lines.append(replace_class_token(line, action.output_class_id or 0))
            else:
                output_lines.append(line)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(output_lines), encoding="utf-8", newline="")
    if not audit.labels:
        raise ValueError(f"No labels were audited in {audit.name}")


def verify_split_uncached(dataset_root: Path, split_name: str, class_count: int) -> dict[str, Any]:
    image_root = dataset_root / "images" / split_name
    label_root = dataset_root / "labels" / split_name
    images = list_images(image_root)
    labels_by_stem = normalized_stems(
        sorted(label_root.rglob("*.txt"), key=lambda path: str(path).casefold()),
        label_root,
    )
    image_map = normalized_stems(images, image_root)
    if set(image_map) != set(labels_by_stem):
        raise ValueError(f"Uncached Ultralytics verification: {split_name} image/label mismatch")

    results = []
    after_counts: Counter[int] = Counter()
    for stem, image_path in image_map.items():
        label_path = labels_by_stem[stem]
        verification = verify_image_label(
            (
                str(image_path),
                str(label_path),
                f"{split_name}: ",
                False,
                class_count,
                0,
                2,
                False,
            )
        )
        results.append(verification)
        if verification[0] is None:
            continue
        verified_labels = verification[1]
        for class_value in verified_labels[:, 0]:
            after_counts[int(class_value)] += 1

    missing, found, empty, corrupt = (
        sum(int(result[index]) for result in results) for index in (5, 6, 7, 8)
    )
    messages = [
        str(result[9]).replace(str(dataset_root), "{dataset}")
        for result in results
        if result[9]
    ]
    return {
        "images": len(images),
        "label_files": len(labels_by_stem),
        "annotation_rows": sum(after_counts.values()),
        "class_counts": {str(key): after_counts[key] for key in sorted(after_counts)},
        "missing": missing,
        "found": found,
        "empty": empty,
        "corrupt": corrupt,
        "messages": messages,
    }


def write_yaml(config: dict[str, Any], output_root: Path) -> bytes:
    updated = dict(config)
    updated["path"] = str(output_root.resolve())
    updated["train"] = "images/train"
    updated["val"] = "images/val"
    return yaml.safe_dump(
        updated, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).encode("utf-8")


def build_exclusion_csv(audits: list[SplitAudit]) -> bytes:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "split",
            "source_label_file",
            "line_number",
            "original_class_id",
            "original_class_name",
            "excluded_row",
            "reason",
        ]
    )
    for audit in audits:
        for label_path, actions in audit.actions.items():
            for action in actions:
                if action.action != "excluded":
                    continue
                writer.writerow(
                    [
                        audit.name,
                        str(label_path),
                        action.line_number,
                        action.class_id,
                        action.class_name,
                        action.row_text,
                        "Class is excluded from the configured eight-class task",
                    ]
                )
    return output.getvalue().encode("utf-8-sig")


def build_report(
    source: Path,
    output: Path,
    original_names: list[str],
    yaml_names: dict[int, str],
    audits: list[SplitAudit],
    verification: dict[str, dict[str, Any]],
    omitted_caches: list[str],
) -> bytes:
    lines = [
        "# YOLO dataset repair report",
        "",
        f"- Source dataset: `{source}`",
        f"- Repaired dataset: `{output}`",
        "- Repair scope: train and validation labels; original dataset is unchanged.",
        "- Method: IDs 8, 9, and 11 are remapped only after per-row source XML "
        "class-name and bounding-box verification.",
        "- Excluded source classes: IDs 5 (`corner`), 6 (`fragment`), "
        "7 (`scratch`), and 10 (`printing_error`).",
        "",
        "## Before/after annotation counts",
        "",
        "| Split | Original ID/name | Before | Repaired ID/name | After | Action |",
        "|---|---|---:|---|---:|---|",
    ]
    combined_before: Counter[int] = Counter()
    combined_after: Counter[int] = Counter()
    for audit in audits:
        combined_before.update(audit.before_counts)
        combined_after.update(audit.after_counts)
        for original_id, name in enumerate(original_names):
            before = audit.before_counts[original_id]
            output_id = CLASS_REMAP.get(original_id, original_id)
            if original_id in EXCLUDED_IDS:
                action = "excluded"
                output_name = "—"
                after = 0
            elif original_id in CLASS_REMAP:
                action = f"remapped {original_id} → {output_id}"
                output_name = yaml_names[output_id]
                after = audit.after_counts[output_id]
            else:
                action = "preserved"
                output_name = yaml_names[output_id]
                after = audit.after_counts[output_id]
            lines.append(
                f"| {audit.name} | {original_id} / `{name}` | {before} | "
                f"{output_id if output_id in range(8) else '—'} / `{output_name}` | "
                f"{after} | {action} |"
            )
    lines.extend(
        [
            "",
            "### Combined split totals",
            "",
            "| Repaired class ID | Class name | Before (source ID) | After |",
            "|---:|---|---:|---:|",
        ]
    )
    for output_id, output_name in sorted(yaml_names.items()):
        source_id = next(
            (old for old, new in CLASS_REMAP.items() if new == output_id), output_id
        )
        lines.append(
            f"| {output_id} | `{output_name}` | {combined_before[source_id]} "
            f"(source ID {source_id}) | {combined_after[output_id]} |"
        )

    remapped_by_split = {
        audit.name: sum(
            1
            for actions in audit.actions.values()
            for action in actions
            if action.action == "remapped"
        )
        for audit in audits
    }
    excluded_by_split = {
        audit.name: sum(
            1
            for actions in audit.actions.values()
            for action in actions
            if action.action == "excluded"
        )
        for audit in audits
    }
    total_remapped = sum(remapped_by_split.values())
    total_excluded = sum(excluded_by_split.values())
    lines.extend(
        [
            "",
            "## Repair totals",
            "",
            f"- Remapped annotation rows: **{total_remapped}** "
            f"({', '.join(f'{split}: {count}' for split, count in remapped_by_split.items())}).",
            f"- Excluded annotation rows: **{total_excluded}** "
            f"({', '.join(f'{split}: {count}' for split, count in excluded_by_split.items())}).",
        ]
    )
    for class_id in sorted(EXCLUDED_IDS):
        rows = sum(audit.before_counts[class_id] for audit in audits)
        lines.append(
            f"- Excluded original ID {class_id} (`{original_names[class_id]}`): {rows} rows."
        )

    lines.extend(["", "## Uncached Ultralytics verification", ""])
    lines.extend(
        [
            "| Split | Images | Label files | Annotation rows | Missing | Empty | Corrupt |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, result in verification.items():
        lines.append(
            f"| {split} | {result['images']} | {result['label_files']} | "
            f"{result['annotation_rows']} | {result['missing']} | "
            f"{result['empty']} | {result['corrupt']} |"
        )
    total_corrupt = sum(result["corrupt"] for result in verification.values())
    total_images = sum(result["images"] for result in verification.values())
    lines.extend(
        [
            "",
            f"Result: **{total_images} images verified; {total_corrupt} corrupt samples.** "
            "Verification used Ultralytics `verify_image_label` directly for each "
            "image/label pair, without reading or creating a dataset cache.",
            "",
            "## Excluded cache files and unresolved issues",
            "",
            "Generated source cache files were not copied: "
            + (", ".join(f"`{item}`" for item in omitted_caches) if omitted_caches else "none found")
            + ".",
            "Unresolved class IDs, malformed rows, missing image/label pairs, "
            "unreadable source annotations, and unverified source boxes: **none**. "
            "All excluded rows are listed in `excluded_annotations.csv`.",
            "",
        ]
    )
    return ("\n".join(lines)).encode("utf-8")


def output_inventory(root: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name != "repair_manifest.json"),
        key=lambda p: str(p).casefold(),
    ):
        entries[path.relative_to(root).as_posix()] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return entries


def write_exactly_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to overwrite an existing file with different content: {path}"
            )
        return
    with path.open("xb") as stream:
        stream.write(content)


def main() -> int:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Safely create an independent repaired eight-class YOLO dataset copy."
    )
    parser.add_argument("--source", type=Path, default=project_root / "yolo_dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "yolo_dataset_repaired_8class",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=project_root
        / "solar_cell_EL_image"
        / "PVELAD"
        / "EL2021"
        / "annotation_classes.txt",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=project_root
        / "solar_cell_EL_image"
        / "PVELAD"
        / "EL2021"
        / "trainval"
        / "Annotations",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Audit both splits and print planned changes without creating or copying files.",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    classes_file = args.classes.expanduser().resolve()
    annotations = args.annotations.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output directory must be separate from and outside the source dataset")
    if not classes_file.is_file() or not annotations.is_dir():
        raise FileNotFoundError("Original class list or source XML annotation directory is missing")

    config, original_names, yaml_names = load_source_configuration(source, classes_file)
    audits = [
        audit_split(split, source, annotations, original_names)
        for split in ("train", "val")
    ]
    source_fingerprint, source_inventory = inventory_fingerprint(source)
    omitted_caches = [
        entry["path"]
        for entry in source_inventory
        if Path(entry["path"]).suffix.lower() == ".cache"
    ]
    excluded_csv = build_exclusion_csv(audits)
    exclusion_count = sum(
        1
        for audit in audits
        for actions in audit.actions.values()
        for action in actions
        if action.action == "excluded"
    )
    if exclusion_count != len(list(csv.reader(excluded_csv.decode("utf-8-sig").splitlines()))) - 1:
        raise AssertionError("Excluded annotation CSV row count does not match audit")

    if args.plan_only:
        print(f"Read-only preflight passed for {source}")
        for audit in audits:
            remapped = sum(
                action.action == "remapped"
                for group in audit.actions.values()
                for action in group
            )
            excluded = sum(
                action.action == "excluded"
                for group in audit.actions.values()
                for action in group
            )
            print(
                f"{audit.name}: {len(audit.images)} images, {len(audit.labels)} label files, "
                f"{sum(audit.before_counts.values())} rows; "
                f"remap={remapped}, exclude={excluded}, "
                f"remaining={sum(audit.after_counts.values())}"
            )
        print(f"Generated cache files to omit: {omitted_caches}")
        return 0

    report_path = project_root / "yolo_dataset_repair_report.md"
    excluded_path = project_root / "excluded_annotations.csv"
    stage = output.with_name(output.name + ".building")
    manifest_path = output / "repair_manifest.json"
    if stage.exists():
        raise FileExistsError(
            f"Staging directory already exists; inspect it without deleting it: {stage}"
        )

    if output.exists():
        if not manifest_path.is_file():
            raise FileExistsError(
                f"Output exists without a repair manifest; refusing to overwrite: {output}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("script_version") != SCRIPT_VERSION
            or manifest.get("source") != str(source)
            or manifest.get("source_fingerprint") != source_fingerprint
        ):
            raise ValueError(
                "Existing output does not match this source/script version; "
                "refusing to overwrite it."
            )
        actual_inventory = output_inventory(output)
        if actual_inventory != manifest.get("output_inventory"):
            raise ValueError(
                "Existing repaired output differs from its manifest; refusing to overwrite."
            )
        verification = {
            split: verify_split_uncached(output, split, len(yaml_names))
            for split in ("train", "val")
        }
        if verification != manifest.get("verification"):
            raise ValueError(
                "Existing repaired dataset no longer passes the recorded uncached "
                "verification; refusing to overwrite."
            )
        report = build_report(
            source,
            output,
            original_names,
            yaml_names,
            audits,
            verification,
            omitted_caches,
        )
        if hashlib.sha256(report).hexdigest() != manifest.get("report_sha256"):
            raise ValueError("Recreated report does not match the existing repair manifest.")
        if hashlib.sha256(excluded_csv).hexdigest() != manifest.get("excluded_csv_sha256"):
            raise ValueError("Recreated excluded-row CSV does not match the repair manifest.")
        write_exactly_once(report_path, report)
        write_exactly_once(excluded_path, excluded_csv)
        print(f"Already repaired and verified; no files changed: {output}")
        print(f"Uncached verification: {sum(v['images'] for v in verification.values())} images, "
              f"{sum(v['corrupt'] for v in verification.values())} corrupt samples")
        return 0

    source_bytes = sum(entry["size"] for entry in source_inventory)
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < source_bytes:
        raise OSError(
            f"Insufficient free space for a dataset copy: need at least {source_bytes} bytes, "
            f"available {free_bytes} bytes"
        )

    shutil.copytree(
        source,
        stage,
        ignore=shutil.ignore_patterns("*.cache"),
        copy_function=shutil.copy2,
    )
    omitted_relatives = set(omitted_caches)
    transformed_labels = {
        label.relative_to(source).as_posix()
        for audit in audits
        for label in audit.labels
    }
    for entry in source_inventory:
        relative = entry["path"]
        if (
            relative in omitted_relatives
            or relative == "data.yaml"
            or relative in transformed_labels
        ):
            continue
        source_file = source / Path(relative)
        copied_file = stage / Path(relative)
        if (
            not copied_file.is_file()
            or source_file.stat().st_size != copied_file.stat().st_size
            or sha256_file(source_file) != sha256_file(copied_file)
        ):
            raise OSError(f"Copied file content differs from source: {relative}")

    for audit in audits:
        apply_split_repair(audit, source, stage)

    yaml_content = write_yaml(config, output)
    (stage / "data.yaml").write_bytes(yaml_content)
    verification = {
        split: verify_split_uncached(stage, split, len(yaml_names))
        for split in ("train", "val")
    }
    if any(result["corrupt"] or result["missing"] for result in verification.values()):
        raise ValueError(
            "Uncached Ultralytics verification failed; staging copy left in place for "
            f"inspection: {stage}; results={verification}"
        )
    for split, result in verification.items():
        if result["images"] != len(next(a for a in audits if a.name == split).images):
            raise AssertionError(f"Image count changed in {split}")
        if result["label_files"] != len(next(a for a in audits if a.name == split).labels):
            raise AssertionError(f"Label-file count changed in {split}")
        if set(map(int, result["class_counts"])) - set(range(len(yaml_names))):
            raise AssertionError(f"Out-of-range class IDs remain in {split}")
        expected_counts = next(a for a in audits if a.name == split).after_counts
        if result["class_counts"] != {
            str(class_id): expected_counts[class_id]
            for class_id in sorted(expected_counts)
            if expected_counts[class_id]
        }:
            raise AssertionError(
                f"Ultralytics-loaded annotation counts differ from audited counts in {split}: "
                f"{result['class_counts']} != {dict(expected_counts)}"
            )

    report = build_report(
        source,
        output,
        original_names,
        yaml_names,
        audits,
        verification,
        omitted_caches,
    )
    manifest = {
        "script_version": SCRIPT_VERSION,
        "source": str(source),
        "source_fingerprint": source_fingerprint,
        "source_inventory": source_inventory,
        "class_mapping": {str(key): value for key, value in CLASS_REMAP.items()},
        "excluded_original_class_ids": sorted(EXCLUDED_IDS),
        "verification": verification,
        "report_sha256": hashlib.sha256(report).hexdigest(),
        "excluded_csv_sha256": hashlib.sha256(excluded_csv).hexdigest(),
    }
    manifest["output_inventory"] = output_inventory(stage)
    write_exactly_once(
        stage / "repair_manifest.json",
        (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode("utf-8"),
    )
    stage.rename(output)

    write_exactly_once(report_path, report)
    write_exactly_once(excluded_path, excluded_csv)
    print(f"Repaired dataset: {output}")
    print(f"Dataset YAML: {output / 'data.yaml'}")
    print(f"Excluded annotation report: {excluded_path}")
    print(f"Before/after report: {report_path}")
    print(f"Uncached Ultralytics verification: {sum(v['images'] for v in verification.values())} images, "
          f"{sum(v['corrupt'] for v in verification.values())} corrupt samples")
    print(f"Remapped rows: {sum(a.action == 'remapped' for audit in audits for group in audit.actions.values() for a in group)}")
    print(f"Excluded rows: {exclusion_count}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Repair failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
