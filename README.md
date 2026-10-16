# HabitatSignal

**HabitatSignal** is a local-first wildlife observation prototype. It reads a fixed camera, a video file, or a direct camera stream, detects selected animals with a pretrained computer-vision model, tracks their appearances, and saves a reviewable observation log.

The hackathon prototype focuses on one honest outcome: **showing which supported wildlife labels appeared in this camera view and when**. It does not claim to measure animal populations, diagnose wildlife health, score habitat quality, or recommend interventions without expert review.

## What it does

- Reads a local webcam, existing video file, or direct RTSP/HTTP video stream.
- Runs a pretrained Ultralytics YOLO model and ByteTrack on the video frames.
- Shows annotated detections in a browser dashboard.
- Logs one sighting when each track first appears in a monitoring session.
- Summarizes active tracks, sightings by species, timestamps, and confidence.
- Stores observations locally in SQLite.

The initial model supports these COCO labels: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, and giraffe. Species outside those classes will not be identified by this version.

## Quick start

Python 3.10 or newer is recommended. From the project directory:

```bash
python -m venv .venv
```

Activate the environment, then install and run:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser. The first run downloads the selected model weights (`yolo26n.pt` by default), so allow time for the download and keep an internet connection available during setup.

### Choose a source

- `0` — default local webcam
- `1`, `2`, … — another local camera index
- `path/to/wildlife.mp4` — an existing video file
- `rtsp://...` — a direct RTSP camera stream
- `https://...` — a direct HTTP video or MJPEG stream, when the camera provider permits it

The public viewing page for a nature webcam is often an embedded player, not a direct video stream. HabitatSignal cannot read frames from an arbitrary web page or bypass a provider's access controls. Use a provider-approved direct stream, a local camera, or a video file you are allowed to analyze. For a hackathon demo, test the selected source and prepare a permitted wildlife clip as a fallback.

## Demo plan

1. Start with a prepared wildlife clip to confirm that the model and dashboard work.
2. Use a fixed camera scene and show boxes, labels, and track IDs on detected animals.
3. Show the session's sighting log and explain that a detection is a camera-view observation, not a distinct animal or a population count.
4. If a direct stream is available and authorized, switch the source to the live feed.

## Configuration

Set environment variables before starting the server if needed:

| Variable | Default | Purpose |
|---|---|---|
| `HABITATSIGNAL_MODEL` | `yolo26n.pt` | Ultralytics model checkpoint; larger models trade speed for accuracy |
| `HABITATSIGNAL_DB` | `data/observations.sqlite3` | Local SQLite observation database |
| `PORT` | `5000` | Local web server port |

The server binds to `127.0.0.1` by default. Keep it local during development; do not expose it to the public internet without adding authentication and network safeguards.

## How to interpret observations

The displayed confidence is the detector's confidence for that frame. A track can be missed, split into multiple IDs, or assigned the wrong label. Lighting, distance, occlusion, camera movement, and stream compression can affect results. Review detections against the original footage before using them in ecological analysis. For reliable population or habitat conclusions, collect a defined sampling design and validate it with field experts.

## Development

Run the lightweight store and class-map tests with:

```bash
python -m unittest discover -s tests -v
```

The project structure is intentionally small:

```text
app.py                    Flask API and dashboard routes
habitatsignal/monitor.py  camera worker, tracking, and annotated frames
habitatsignal/database.py SQLite observation store
habitatsignal/species.py  supported model classes
templates/index.html      responsive live dashboard
tests/                    core behavior tests
```

## Attribution and model terms

This project is released under the GNU Affero General Public License v3.0; see `LICENSE`. It uses the Ultralytics Python package and pretrained model weights, which have their own terms. Review the current [Ultralytics license and model terms](https://www.ultralytics.com/license) before redistribution or commercial use. The wildlife monitoring workflow was informed by the open-source [Backyard Critter Cam](https://github.com/unclemattmakes/backyard-critter-cam); this repository contains a separate implementation. The Backyard Critter Cam repository is also licensed under AGPL-3.0.

HabitatSignal is an educational prototype. It is not a substitute for conservation science, wildlife professionals, or environmental authorities.
