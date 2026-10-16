"""Background camera reader and small-insect detection worker."""

from __future__ import annotations

import threading
import time
import uuid
import os
from pathlib import Path
from typing import Any, Iterator

import cv2

from habitatsignal.database import ObservationStore
from habitatsignal.species import model_class_ids, names_by_id
from habitatsignal.sources import is_youtube_url, resolve_youtube_stream


class WildlifeMonitor:
    """Read a local or network video source and expose annotated frames."""

    def __init__(
        self,
        store: ObservationStore,
        model_path: str = "models/fruitfly.pt",
        image_size: int | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        max_detections: int | None = None,
    ):
        self.store = store
        self.model_path = model_path
        self.image_size = image_size or int(os.getenv("HABITATSIGNAL_IMGSZ", "1280"))
        self.confidence = confidence if confidence is not None else float(os.getenv("HABITATSIGNAL_CONF", "0.12"))
        self.iou = iou if iou is not None else float(os.getenv("HABITATSIGNAL_IOU", "0.5"))
        self.max_detections = max_detections or int(os.getenv("HABITATSIGNAL_MAX_DET", "500"))
        self.tracker_config = str(
            Path(__file__).resolve().parent.parent / "trackers" / "bytetrack-fruitfly.yaml"
        )
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_jpeg: bytes | None = None
        self._frame_version = 0
        self._status = "idle"
        self._error: str | None = None
        self._run_id: str | None = None
        self._source_label: str | None = None
        self._active_tracks: dict[str, dict[str, Any]] = {}
        self._class_names: dict[int, str] = {}
        self._observed_track_ids: set[str] = set()
        self._total_frames = 0
        self._last_frame_at: str | None = None

    @staticmethod
    def _source_value(source: str) -> int | str:
        if source.isdecimal():
            return int(source)
        return source

    def start(self, source: str) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"ok": False, "error": "A monitoring session is already running."}
            if not (source.isdecimal() or source.startswith(("rtsp://", "rtsps://", "http://", "https://")) or Path(source).is_file()):
                return {
                    "ok": False,
                    "error": "Use a camera index, existing video file, direct RTSP/HTTP stream, or public YouTube video/live URL.",
                }
            self._stop_event.clear()
            self._run_id = str(uuid.uuid4())
            self._source_label = f"Camera {source}" if source.isdecimal() else Path(source).name if "://" not in source else "Network camera"
            self._status = "starting"
            self._error = None
            self._latest_jpeg = None
            self._active_tracks = {}
            self._observed_track_ids = set()
            self._total_frames = 0
            self._last_frame_at = None
            self._thread = threading.Thread(
                target=self._run, args=(self._source_value(source), self._run_id), daemon=True
            )
            self._thread.start()
            return {"ok": True, "status": self._status, "run_id": self._run_id}

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._status not in {"idle", "error"}:
                self._status = "stopping"

    def state(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for track in self._active_tracks.values():
                counts[track["species"]] = counts.get(track["species"], 0) + 1
            return {
                "status": self._status,
                "error": self._error,
                "run_id": self._run_id,
                "source": self._source_label,
                "frames_processed": self._total_frames,
                "last_frame_at": self._last_frame_at,
                "active_tracks": list(self._active_tracks.values()),
                "active_by_species": counts,
                "model": self.model_path,
            }

    def _run(self, source: int | str, run_id: str) -> None:
        capture = None
        try:
            from ultralytics import YOLO

            if not Path(self.model_path).is_file() and Path(self.model_path).name == "fruitfly.pt":
                raise RuntimeError(
                    "The fruit-fly model is not trained yet. Follow the Training section in README.md, "
                    "then set HABITATSIGNAL_MODEL to the trained checkpoint."
                )
            model = YOLO(self.model_path)
            capture_source = resolve_youtube_stream(source) if isinstance(source, str) and is_youtube_url(source) else source
            capture = cv2.VideoCapture(capture_source)
            if not capture.isOpened():
                raise RuntimeError("Could not open this source. Check that the camera, file, or public stream is available.")
            with self._lock:
                self._status = "running"

            self._class_names = names_by_id(model.names)
            class_ids = model_class_ids(self._class_names)
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    if isinstance(source, str):
                        break
                    time.sleep(0.1)
                    continue

                result = model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_config,
                    classes=class_ids,
                    conf=self.confidence,
                    iou=self.iou,
                    imgsz=self.image_size,
                    max_det=self.max_detections,
                    verbose=False,
                )[0]
                self._record_tracks(result, run_id)
                annotated = result.plot()
                cv2.putText(
                    annotated,
                    "HabitatSignal  |  tracked fly appearances, not an exact fly count",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (236, 246, 238),
                    2,
                    cv2.LINE_AA,
                )
                encoded_ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if encoded_ok:
                    with self._lock:
                        self._latest_jpeg = encoded.tobytes()
                        self._frame_version += 1
                        self._total_frames += 1
                        self._last_frame_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                        self._frame_condition.notify_all()

            with self._lock:
                self._status = "idle"
        except Exception as exc:  # Surface source and model errors in the dashboard.
            with self._lock:
                self._status = "error"
                self._error = str(exc)
        finally:
            if capture is not None:
                capture.release()

    def _record_tracks(self, result: Any, run_id: str) -> None:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return
        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.float().cpu().tolist()
        xyxy = boxes.xyxy.int().cpu().tolist()
        now = time.strftime("%H:%M:%S UTC", time.gmtime())

        with self._lock:
            current_tracks: dict[str, dict[str, Any]] = {}
            for track_id, class_id, confidence, box in zip(track_ids, class_ids, confidences, xyxy):
                species = self._class_names.get(int(class_id))
                if species is None:
                    continue
                key = str(track_id)
                previous = self._active_tracks.get(key)
                if key not in self._observed_track_ids:
                    self.store.record(run_id, key, species, confidence)
                    self._observed_track_ids.add(key)
                current_tracks[key] = {
                    "track_id": key,
                    "species": species,
                    "first_seen": previous["first_seen"] if previous else now,
                    "last_seen": now,
                    "confidence": max(previous["confidence"], round(float(confidence), 3)) if previous else round(float(confidence), 3),
                    "box": box,
                }
            # This view is only what's visible in the latest frame, not all session visitors.
            self._active_tracks = current_tracks

    def mjpeg_stream(self) -> Iterator[bytes]:
        with self._lock:
            last_version = self._frame_version
        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(
                    lambda: self._frame_version > last_version or self._status in {"idle", "error"},
                    timeout=10,
                )
                frame = self._latest_jpeg
                version = self._frame_version
                status = self._status
            if frame and version > last_version:
                last_version = version
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if status in {"idle", "error"}:
                return

