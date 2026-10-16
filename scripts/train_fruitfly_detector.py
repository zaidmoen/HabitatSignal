"""Fine-tune a small-object detector for fruit flies near produce."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--model", default="yolo26s.pt", help="Pretrained detector to fine-tune")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics estimate a safe batch")
    parser.add_argument("--device", default=None, help="Examples: 0, cpu, or mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="fruitfly-small-object")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    try:
        import yaml
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install project dependencies first: python -m pip install -r requirements.txt") from exc

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    class_names = data.get("names", {})
    if isinstance(class_names, dict):
        class_names = list(class_names.values())
    if not any("fruitfly" in str(name).lower().replace(" ", "") for name in class_names):
        raise SystemExit("Dataset YAML must define a class named fruitfly")

    model = YOLO(args.model)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": 25,
        "workers": args.workers,
        "project": str(ROOT / "runs" / "detect"),
        "name": args.name,
        "exist_ok": True,
        "close_mosaic": 10,
        "seed": 42,
        "plots": True,
    }
    if args.device:
        train_args["device"] = args.device

    model.train(**train_args)
    best_checkpoint = Path(model.trainer.best)
    if not best_checkpoint.is_file():
        raise SystemExit(f"Training finished without a best checkpoint: {best_checkpoint}")

    # Keep a real held-out test split when one is configured.
    split = "test" if data.get("test") else "val"
    best_model = YOLO(str(best_checkpoint))
    best_model.val(data=str(data_path), split=split, imgsz=args.imgsz, plots=True)

    target = ROOT / "models" / "fruitfly.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, target)
    print(f"\nReady for HabitatSignal: {target}")
    print("Set HABITATSIGNAL_MODEL to this checkpoint if you move it elsewhere.")


if __name__ == "__main__":
    main()
