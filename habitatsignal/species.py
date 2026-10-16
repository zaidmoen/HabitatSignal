"""Class-name helpers for pretrained and project-trained detectors."""

from __future__ import annotations


# COCO class IDs from the pretrained Ultralytics detector.
COCO_WILDLIFE_CLASS_IDS = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

# Kept as an alias for existing store rows and tests from the wildlife prototype.
WILDLIFE_CLASS_IDS = COCO_WILDLIFE_CLASS_IDS


def names_by_id(names: dict | list | tuple) -> dict[int, str]:
    """Normalize Ultralytics class names to an integer-keyed mapping."""
    pairs = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(class_id): str(name) for class_id, name in pairs}


def model_class_ids(names: dict[int, str] | None = None) -> list[int]:
    """Return all classes in the loaded detector, or legacy wildlife classes."""
    return sorted(names if names is not None else COCO_WILDLIFE_CLASS_IDS)


def label_for(class_id: int) -> str | None:
    """Return a supported common-name label, if this class is wildlife."""
    return WILDLIFE_CLASS_IDS.get(int(class_id))

