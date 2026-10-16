"""HabitatSignal local wildlife observation dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from habitatsignal.database import ObservationStore
from habitatsignal.monitor import WildlifeMonitor


ROOT = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.environ.get("HABITATSIGNAL_DB", ROOT / "data" / "observations.sqlite3"))
MODEL_PATH = os.environ.get("HABITATSIGNAL_MODEL", "yolo26n.pt")

app = Flask(__name__)
store = ObservationStore(DATABASE_PATH)
monitor = WildlifeMonitor(store=store, model_path=MODEL_PATH)


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "habitatsignal"})


@app.post("/api/start")
def start_monitoring():
    payload = request.get_json(silent=True) or {}
    source = str(payload.get("source", "")).strip()
    if not source:
        return jsonify({"error": "Enter a camera index, video path, or direct stream URL."}), 400
    if len(source) > 500:
        return jsonify({"error": "The video source is too long."}), 400

    result = monitor.start(source)
    if not result["ok"]:
        return jsonify({"error": result["error"]}), 409
    return jsonify(result), 202


@app.post("/api/stop")
def stop_monitoring():
    monitor.stop()
    return jsonify({"status": "stopping"})


@app.get("/api/state")
def current_state():
    state = monitor.state()
    state["summary"] = store.summary(state.get("run_id"))
    return jsonify(state)


@app.get("/api/history")
def observation_history():
    return jsonify(store.recent(limit=100))


@app.get("/video_feed")
def video_feed():
    return Response(
        monitor.mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), threaded=True)
