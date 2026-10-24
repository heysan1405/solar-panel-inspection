from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PIL_FORMAT_SUFFIXES = {
    "BMP": {".bmp"},
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "TIFF": {".tif", ".tiff"},
    "WEBP": {".webp"},
}


@dataclass
class Finding:
    category: str
    problem: str
    repairability: str
    detail: str
    path: Path


@dataclass
class ImageAudit:
    image_path: Path
    label_path: Path
    image_readable: bool = True
    label_valid: bool = True
    label_missing: bool = False
    manual_review: bool = False
    safe_repair: bool = False
    findings: list[Finding] = field(default_factory=list)

    def add(
        self,
        category: str,
        problem: str,
        detail: str,
        path: Path,
        *,
        repairability: str = "No; manual inspection required",
        manual_review: bool = True,
    ) -> None:
        self.findings.append(Finding(category, problem, repairability, detail, path))
        self.manual_review |= manual_review


def resolve_dataset_root(config: dict[str, Any], yaml_path: Path) -> Path:
    configured_root = config.get("path")
    if configured_root is None:
        return yaml_path.parent.resolve()
    root = Path(str(configured_root)).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def resolve_split_images(
    split_value: Any, dataset_root: Path, yaml_path: Path
) -> list[Path]:
    values = split_value if isinstance(split_value, list) else [split_value]
    result: list[Path] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"Unsupported validation path value: {value!r}")
        split_path = Path(value).expanduser()
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        split_path = split_path.resolve()
        if split_path.is_file() and split_path.suffix.lower() == ".txt":
            for line in split_path.read_text(encoding="utf-8-sig").splitlines():
                entry = line.strip()
                if not entry:
                    continue
                image_path = Path(entry).expanduser()
                if not image_path.is_absolute():
                    image_path = split_path.parent / image_path
                result.append(image_path.resolve())
        elif split_path.is_dir():
            result.extend(
                path.resolve()
                for path in split_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            raise FileNotFoundError(
                f"Validation path from {yaml_path} does not exist: {split_path}"
            )
    return sorted(set(result), key=lambda path: str(path).casefold())


def corresponding_label(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indexes = [i for i, part in enumerate(parts) if part.casefold() == "images"]
    if not image_indexes:
        raise ValueError(
            f"Cannot map image path to labels because it has no 'images' folder: {image_path}"
        )
    parts[image_indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt").resolve()


def check_image(audit: ImageAudit) -> None:
    path = audit.image_path
    try:
        with Image.open(path) as image:
            image_format = image.format
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"invalid dimensions: {image.width}x{image.height}")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        audit.image_readable = False
        audit.add(
            "corrupt_image",
            "Image cannot be fully decoded by Pillow",
            f"{type(exc).__name__}: {exc}",
            path,
        )
        return

    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise ValueError("OpenCV imdecode returned no image")
    except (OSError, cv2.error, ValueError) as exc:
        audit.image_readable = False
        audit.add(
            "corrupt_image",
            "Image cannot be decoded by OpenCV (Ultralytics-compatible check)",
            f"{type(exc).__name__}: {exc}",
            path,
        )
        return

    expected_suffixes = PIL_FORMAT_SUFFIXES.get(image_format or "")
    if expected_suffixes and path.suffix.lower() not in expected_suffixes:
        audit.add(
            "encoding_mismatch",
            "Image encoding does not match its filename extension",
            f"Pillow detected {image_format}; filename extension is {path.suffix}",
            path,
            repairability="Potentially repairable by re-encoding a copy",
        )


def check_label(audit: ImageAudit, class_count: int) -> None:
    path = audit.label_path
    if not path.is_file():
        audit.label_missing = True
        audit.label_valid = False
        audit.add(
            "missing_label",
            "Image has no corresponding YOLO label file",
            f"Expected label path: {path}",
            audit.image_path,
        )
        return

    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        audit.label_valid = False
        audit.add(
            "unreadable_label",
            "Label file cannot be read as UTF-8 text",
            f"{type(exc).__name__}: {exc}",
            path,
        )
        return

    rows: list[tuple[int, int, float, float, float, float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            audit.label_valid = False
            audit.add(
                "malformed_label",
                "YOLO detection row must contain exactly five values",
                f"Line {line_number} has {len(fields)} fields: {line!r}",
                path,
            )
            continue

        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            audit.label_valid = False
            audit.add(
                "non_numeric_label",
                "Label row contains a non-numeric value",
                f"Line {line_number}: {line!r}; {type(exc).__name__}: {exc}",
                path,
            )
            continue

        if not all(math.isfinite(value) for value in values):
            audit.label_valid = False
            audit.add(
                "non_finite_label",
                "Label row contains a non-finite value",
                f"Line {line_number}: {line!r}",
                path,
            )
            continue

        class_value, x_center, y_center, width, height = values
        if not class_value.is_integer():
            audit.label_valid = False
            audit.add(
                "invalid_class_id",
                "Class ID is not an integer",
                f"Line {line_number}: class ID is {class_value}",
                path,
            )
        elif not 0 <= int(class_value) < class_count:
            audit.label_valid = False
            audit.add(
                "class_id_out_of_range",
                f"Class ID {int(class_value)} is outside the configured range 0-{class_count - 1}",
                f"Line {line_number}: {line!r}",
                path,
            )

        coordinates = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }
        out_of_range = {
            name: value
            for name, value in coordinates.items()
            if not 0.0 <= value <= 1.0
        }
        if out_of_range:
            audit.label_valid = False
            audit.add(
                "coordinate_out_of_range",
                "Bounding-box value is outside normalized range [0, 1]",
                f"Line {line_number}: {out_of_range}",
                path,
            )
        if width == 0.0 or height == 0.0:
            audit.label_valid = False
            audit.add(
                "zero_area_box",
                "Bounding box has zero width or height",
                f"Line {line_number}: width={width}, height={height}",
                path,
            )
        if (
            width > 0.0
            and height > 0.0
            and not out_of_range
            and (
                x_center - width / 2 < 0
                or y_center - height / 2 < 0
                or x_center + width / 2 > 1
                or y_center + height / 2 > 1
            )
        ):
            audit.label_valid = False
            audit.add(
                "box_extends_outside_image",
                "Bounding box extends beyond image boundaries",
                f"Line {line_number}: {line!r}",
                path,
            )
        rows.append(
            (line_number, int(class_value) if class_value.is_integer() else -1,
             x_center, y_center, width, height)
        )

    seen: dict[tuple[int, float, float, float, float], int] = {}
    boxes_by_coordinates: dict[tuple[float, float, float, float], tuple[int, int]] = {}
    for line_number, class_id, x, y, width, height in rows:
        key = (class_id, x, y, width, height)
        if key in seen:
            audit.label_valid = False
            audit.add(
                "duplicate_label",
                "Exact duplicate annotation row",
                f"Lines {seen[key]} and {line_number} have identical class and box values",
                path,
            )
        else:
            seen[key] = line_number
        coordinates = (x, y, width, height)
        previous = boxes_by_coordinates.get(coordinates)
        if previous is not None and previous[0] != class_id:
            audit.label_valid = False
            audit.add(
                "conflicting_duplicate_box",
                "Identical bounding box is assigned to different classes",
                f"Lines {previous[1]} and {line_number} use classes "
                f"{previous[0]} and {class_id}",
                path,
            )
        else:
            boxes_by_coordinates[coordinates] = (class_id, line_number)

    if not text.strip():
        audit.add(
            "empty_label",
            "Empty label file; valid representation of a background image",
            "No annotation rows; Ultralytics treats this as a background image",
            path,
            repairability="Not applicable; legitimate empty label",
            manual_review=False,
        )


def inspect_cache(
    image_paths: list[Path],
    label_paths: list[Path],
    cache_path: Path,
    actual_problem_images: set[Path],
) -> tuple[str, str, tuple[int, ...] | None, list[str]]:
    if not cache_path.is_file():
        return "missing", "No validation cache exists", None, []

    try:
        from ultralytics.data.utils import get_hash

        cache = np.load(cache_path, allow_pickle=True).item()
        cached_hash = cache.get("hash")
        current_hash = get_hash([str(path) for path in label_paths + image_paths])
        hash_matches = cached_hash == current_hash
        results_value = cache.get("results")
        results = tuple(int(value) for value in results_value) if results_value is not None else None
        messages = [
            re.sub(r"\x1b\[[0-9;]*m", "", str(message))
            for message in cache.get("msgs", [])
        ]
        cached_problem_images = {
            image_path
            for image_path in image_paths
            if any(str(image_path) in message for message in messages)
        }
        cache_matches_audit = (
            results is not None
            and len(results) >= 5
            and results[3] == len(actual_problem_images)
            and results[4] == len(image_paths)
            and len(messages) == results[3]
            and cached_problem_images == actual_problem_images
        )
        if not hash_matches:
            hash_status = "STALE: hash does not match current image/label paths and sizes"
        elif not cache_matches_audit:
            hash_status = "STALE: cached scan results do not match the current audit"
        else:
            hash_status = (
                "matches current image/label paths and sizes and current audit findings"
            )
        return hash_status, str(cache.get("version", "unknown")), results, messages
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as exc:
        return (
            "unreadable",
            f"{type(exc).__name__}: {exc}",
            None,
            [],
        )


def write_outputs(
    csv_path: Path,
    summary_path: Path,
    yaml_path: Path,
    image_audits: list[ImageAudit],
    orphan_labels: list[Path],
    cache_status: str,
    cache_version: str,
    cache_results: tuple[int, ...] | None,
    cache_messages: list[str],
    class_names: dict[int, str],
) -> None:
    all_findings: list[Finding] = [
        finding for audit in image_audits for finding in audit.findings
    ]
    for label_path in orphan_labels:
        all_findings.append(
            Finding(
                "missing_image",
                "Label file has no corresponding validation image",
                "No; missing images must not be invented",
                "Label has no image with the matching relative path and stem",
                label_path,
            )
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "category",
                "file_path",
                "image_path",
                "label_path",
                "problem",
                "repairability",
                "detail",
            ]
        )
        for finding in all_findings:
            matching = next(
                (
                    audit
                    for audit in image_audits
                    if finding.path in (audit.image_path, audit.label_path)
                ),
                None,
            )
            writer.writerow(
                [
                    finding.category,
                    str(finding.path),
                    str(matching.image_path) if matching else "",
                    str(matching.label_path) if matching else "",
                    finding.problem,
                    finding.repairability,
                    finding.detail,
                ]
            )

    image_count = len(image_audits)
    readable_count = sum(audit.image_readable for audit in image_audits)
    corrupt_image_count = image_count - readable_count
    missing_label_count = sum(audit.label_missing for audit in image_audits)
    invalid_label_count = sum(
        not audit.label_valid and not audit.label_missing for audit in image_audits
    )
    safe_repair_count = sum(audit.safe_repair for audit in image_audits)
    manual_review_count = sum(audit.manual_review for audit in image_audits)
    affected_samples = sum(bool(audit.findings) for audit in image_audits)
    finding_counts = Counter(finding.category for finding in all_findings)
    invalid_class_counts = Counter(
        int(match.group(1))
        for finding in all_findings
        if finding.category == "class_id_out_of_range"
        and (match := re.search(r"Class ID (\d+)", finding.problem))
    )
    empty_label_count = sum(
        any(finding.category == "empty_label" for finding in audit.findings)
        for audit in image_audits
    )

    lines = [
        "YOLO validation dataset audit",
        f"Dataset YAML: {yaml_path}",
        f"Class IDs from YAML: {', '.join(f'{key}={value}' for key, value in sorted(class_names.items()))}",
        "",
        f"Images inspected: {image_count}",
        f"Valid/readable images: {readable_count}",
        f"Corrupt or unreadable images: {corrupt_image_count}",
        f"Missing labels: {missing_label_count}",
        f"Invalid label files: {invalid_label_count}",
        f"Safely repairable files: {safe_repair_count}",
        f"Files requiring manual inspection: {manual_review_count}",
        f"Validation samples with one or more findings: {affected_samples}",
        f"Legitimate empty labels: {empty_label_count}",
        f"Orphan label files (missing image): {len(orphan_labels)}",
        "Finding rows by category: "
        + (
            ", ".join(f"{name}={count}" for name, count in sorted(finding_counts.items()))
            if finding_counts
            else "(none)"
        ),
        "Out-of-range class IDs by annotation row: "
        + (
            ", ".join(
                f"{class_id}={count}"
                for class_id, count in sorted(invalid_class_counts.items())
            )
            if invalid_class_counts
            else "(none)"
        ),
        "",
        f"Ultralytics cache: {cache_status}",
        f"Cache version or error: {cache_version}",
        f"Cached scan results (found, missing, empty, corrupt, total): {cache_results}",
        f"Cached diagnostic message count: {len(cache_messages)}",
        "",
        "Ultralytics cache messages (recorded at the last cached scan):",
    ]
    lines.extend(cache_messages or ["(none)"])
    if cache_status.startswith("STALE") or cache_status == "unreadable":
        lines.extend(
            [
                "",
                "Cache note: invalidate only this validation cache before the next Ultralytics scan.",
            ]
        )
    lines.extend(
        [
            "",
            f"Detailed per-file findings: {csv_path}",
            "No dataset files were modified by this audit.",
        ]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit validation images, YOLO labels, and the split's Ultralytics cache."
    )
    default_yaml = Path(__file__).resolve().parent / "yolo_dataset" / "data.yaml"
    parser.add_argument("--data-yaml", type=Path, default=default_yaml)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    yaml_path = args.data_yaml.expanduser().resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {yaml_path}")
    class_count = int(config["nc"])
    names_value = config.get("names", {})
    if isinstance(names_value, list):
        class_names = dict(enumerate(str(name) for name in names_value))
    elif isinstance(names_value, dict):
        class_names = {int(key): str(value) for key, value in names_value.items()}
    else:
        raise ValueError("Dataset YAML 'names' must be a list or mapping")
    if set(class_names) != set(range(class_count)):
        raise ValueError(
            f"Dataset YAML names do not define exactly class IDs 0-{class_count - 1}: "
            f"{sorted(class_names)}"
        )

    dataset_root = resolve_dataset_root(config, yaml_path)
    image_paths = resolve_split_images(config["val"], dataset_root, yaml_path)
    image_audits: list[ImageAudit] = []
    expected_labels: set[Path] = set()
    for image_path in image_paths:
        label_path = corresponding_label(image_path)
        expected_labels.add(label_path)
        audit = ImageAudit(image_path=image_path, label_path=label_path)
        if not image_path.is_file():
            audit.image_readable = False
            audit.add(
                "missing_image",
                "Validation image path does not exist",
                str(image_path),
                image_path,
            )
        else:
            check_image(audit)
        check_label(audit, class_count)
        image_audits.append(audit)

    label_dirs = {path.parent for path in expected_labels}
    existing_labels = {
        path.resolve()
        for label_dir in label_dirs
        if label_dir.is_dir()
        for path in label_dir.rglob("*.txt")
        if path.is_file()
    }
    orphan_labels = sorted(
        (path for path in existing_labels if path not in expected_labels),
        key=lambda path: str(path).casefold(),
    )

    ordered_labels = [audit.label_path for audit in image_audits]
    actual_problem_images = {
        audit.image_path
        for audit in image_audits
        if not audit.image_readable or not audit.label_valid
    }
    cache_path = (
        ordered_labels[0].parent.with_suffix(".cache")
        if ordered_labels
        else next(iter(label_dirs), dataset_root / "labels" / "val").with_suffix(".cache")
    )
    cache_status, cache_version, cache_results, cache_messages = inspect_cache(
        image_paths, ordered_labels, cache_path, actual_problem_images
    )
    csv_path = (args.csv or yaml_path.parent.parent / "validation_audit.csv").resolve()
    summary_path = (
        args.summary or yaml_path.parent.parent / "validation_audit_summary.txt"
    ).resolve()
    write_outputs(
        csv_path,
        summary_path,
        yaml_path,
        image_audits,
        orphan_labels,
        cache_status,
        cache_version,
        cache_results,
        cache_messages,
        class_names,
    )

    print(f"Images inspected: {len(image_audits)}")
    print(f"Corrupt/unreadable images: {sum(not audit.image_readable for audit in image_audits)}")
    print(f"Missing labels: {sum(audit.label_missing for audit in image_audits)}")
    print(
        "Invalid label files: "
        f"{sum(not audit.label_valid and not audit.label_missing for audit in image_audits)}"
    )
    print(f"Ultralytics cache status: {cache_status}")
    print(f"CSV report: {csv_path}")
    print(f"Summary report: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
