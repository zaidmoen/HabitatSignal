import tempfile
import unittest
from pathlib import Path

from habitatsignal.database import ObservationStore
from habitatsignal.species import WILDLIFE_CLASS_IDS, label_for, model_class_ids, names_by_id
from habitatsignal.sources import is_youtube_url, resolve_youtube_stream


class SpeciesTests(unittest.TestCase):
    def test_only_supported_wildlife_labels_are_exposed(self):
        self.assertEqual(label_for(20), "elephant")
        self.assertIsNone(label_for(2))
        self.assertEqual(model_class_ids(), sorted(WILDLIFE_CLASS_IDS))
        self.assertEqual(names_by_id({0: "fruitfly", "1": "other_fly"}), {0: "fruitfly", 1: "other_fly"})
        self.assertEqual(model_class_ids({0: "fruitfly"}), [0])


class SourceTests(unittest.TestCase):
    def test_youtube_live_and_short_links_are_recognized(self):
        self.assertTrue(is_youtube_url("https://www.youtube.com/live/example"))
        self.assertTrue(is_youtube_url("https://youtu.be/example"))
        self.assertFalse(is_youtube_url("https://youtube.com.example.org/live/example"))
        self.assertFalse(is_youtube_url("file:///tmp/video.mp4"))

    def test_youtube_extractor_returns_playable_url(self):
        import sys
        from unittest.mock import patch

        class FakeDownloader:
            last_options = None

            def __init__(self, options):
                self.options = options
                type(self).last_options = options

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, source, download):
                self.request = (source, download)
                return {"url": "https://media.example/live.m3u8"}

        fake_module = type("FakeYtDlp", (), {"YoutubeDL": FakeDownloader})
        with patch.dict(sys.modules, {"yt_dlp": fake_module}):
            self.assertEqual(
                resolve_youtube_stream("https://www.youtube.com/live/example"),
                "https://media.example/live.m3u8",
            )
        self.assertEqual(FakeDownloader.last_options["format"], "best/bestvideo*")


class StoreTests(unittest.TestCase):
    def test_sightings_are_counted_once_per_track_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ObservationStore(Path(directory) / "observations.sqlite3")
            store.record("run-a", "17", "elephant", 0.91)
            store.record("run-a", "17", "elephant", 0.95)
            store.record("run-a", "18", "bird", 0.79)

            self.assertEqual(store.summary("run-a")["total_visits"], 2)
            self.assertEqual(store.summary("run-a")["by_species"], [
                {"species": "bird", "visits": 1},
                {"species": "elephant", "visits": 1},
            ])
            self.assertEqual(len(store.recent()), 2)


if __name__ == "__main__":
    unittest.main()

