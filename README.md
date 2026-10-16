# HabitatSignal

HabitatSignal is a close-up computer-vision monitor for fruit flies around produce. It detects flies with a custom Ultralytics YOLO model, tracks them between frames, and records camera-view appearances in a local SQLite database.

## Important: train the fruit-fly model first

The older `yolo26n.pt` wildlife model was trained on COCO classes and does **not** recognize fruit flies. HabitatSignal now expects a custom checkpoint at `models/fruitfly.pt`. Build that checkpoint from labeled images of the kind of fruit, camera, and lighting you will use. The app reports a clear setup error if the checkpoint is missing instead of silently pretending to detect flies.

## Prepare the dataset

As a starting point, the public [Fruit-Fly-Detection dataset](https://universe.roboflow.com/keshavs-workspace-sscce/fruit-fly-detection-2rfnl) has a `fruitfly` class and is published under CC BY 4.0. Its publisher reports 1,357 images and 90.2% mAP@50 for a hosted model; those numbers are from that project's own evaluation and do not predict performance on your camera. Review the dataset license and attribution requirements before using or redistributing it.

For a stronger result, add your own footage from the intended setup: the target fruit, camera distance, background, day/night lighting, and flies at different sizes and positions. Include hard negatives such as fruit spots, dust, reflections, and other small insects. Keep every image from one source video or capture session in only one split; putting near-identical frames in both train and test makes the test score misleading.

Export object-detection labels in Ultralytics YOLO format. Draw a tight box around each visible fruit fly. Keep `fruitfly` as the class name for a general detector; do not claim species identification unless an entomologist has verified species-level labels. Include images with no flies as negative examples.

Copy `configs/fruitfly.data.yaml.example` to `configs/fruitfly.data.yaml`, then set `path` to your dataset root. The expected layout is:

```text
fruitfly_dataset/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

Each image should have a matching `.txt` label file; an empty file means no fruit flies are present. Keep the test split separate and only use it for final evaluation.

## Train and run

Python 3.10 or newer is recommended. A CUDA GPU is strongly recommended for 1280-pixel training.

```bash
python -m venv .venv
```

Activate the environment, install dependencies, and train:

```bash
python -m pip install --upgrade -r requirements.txt
python scripts/train_fruitfly_detector.py --data configs/fruitfly.data.yaml
python app.py
```

Training fine-tunes `yolo26s.pt` at 1280 pixels by default, evaluates the held-out test split when configured, and copies the best checkpoint to `models/fruitfly.pt`. On a GPU with less memory, try `--imgsz 960 --batch 4`; if the fly occupies only a few pixels in the camera image, move the camera closer or use a macro lens before lowering image size.

Open `http://127.0.0.1:5000`, connect a close-focus camera/video, or paste a public YouTube Live URL. YouTube pages are resolved with `yt-dlp`; provider access and live-stream availability can change. For camera device indices, use `0`, `1`, and so on. Local video files and direct RTSP/HTTP streams are also supported.

## Camera setup and accuracy

- Fill the frame with the fruit and its immediate surroundings. A wide wildlife webcam usually shows flies too small for dependable detection.
- Lock focus on the surface where flies land. Use even, diffuse light and avoid strong glare, motion blur, and a moving camera.
- Capture representative examples from the exact setup and label them carefully. A general internet dataset is a starting point, not a substitute for local validation.
- Compare validation and test precision/recall, then inspect missed flies and false alarms in the original footage. Adjust confidence only after reviewing those errors.
- The UI counts tracked appearances. A fly can be missed, receive multiple track IDs, or be counted again after leaving and returning; this is not a validated number of unique insects or a pest-risk diagnosis.

Ultralytics supports custom dataset training and configurable image size; see its [training](https://docs.ultralytics.com/modes/train) and [prediction](https://docs.ultralytics.com/modes/predict) guides. The inference defaults in this project are set for small, low-contrast targets and can be changed with environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `HABITATSIGNAL_MODEL` | `models/fruitfly.pt` | Trained detector checkpoint |
| `HABITATSIGNAL_IMGSZ` | `1280` | Inference image size; larger can help tiny targets but uses more memory |
| `HABITATSIGNAL_CONF` | `0.12` | Minimum detection confidence; lower values can increase false positives |
| `HABITATSIGNAL_IOU` | `0.5` | Non-maximum suppression overlap threshold |
| `HABITATSIGNAL_MAX_DET` | `500` | Maximum detections per frame |
| `HABITATSIGNAL_DB` | `data/observations.sqlite3` | Local observation database |
| `PORT` | `5000` | Local web server port |

The server binds to `127.0.0.1` by default. Keep it local during development; do not expose it to the public internet without authentication and network safeguards.

## Development

```bash
python -m unittest discover -s tests -v
```

Project layout:

```text
app.py                              Flask API and dashboard
habitatsignal/monitor.py            camera reader, detection, and tracking
habitatsignal/database.py           SQLite observation store
habitatsignal/sources.py             YouTube stream URL resolver
scripts/train_fruitfly_detector.py  fine-tuning and held-out evaluation
configs/                            example dataset config
trackers/                            small-fly ByteTrack settings
templates/index.html                monitoring dashboard
```

## Attribution and model terms

HabitatSignal is released under the GNU Affero General Public License v3.0; see `LICENSE`. The Roboflow fruit-fly dataset is separately licensed CC BY 4.0 by its publisher. Ultralytics and its pretrained weights have their own terms; review the current [Ultralytics license](https://www.ultralytics.com/license) before redistribution or commercial use.

This project is an educational prototype. It does not replace an entomologist, food-safety inspection, or a validated pest-management system.
