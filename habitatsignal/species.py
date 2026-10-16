"""Species supported by the pretrained COCO object detector."""

from __future__ import annotations


# COCO class IDs from the pretrained Ultralytics detector.
WILDLIFE_CLASS_IDS = {
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


def model_class_ids() -> list[int]:
    """Return the supported detector classes in stable numeric order."""
    return list(WILDLIFE_CLASS_IDS)


def label_for(class_id: int) -> str | None:
    """Return a supported common-name label, if this class is wildlife."""
    return WILDLIFE_CLASS_IDS.get(int(class_id))

