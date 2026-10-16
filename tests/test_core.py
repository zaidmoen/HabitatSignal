import tempfile
import unittest
from pathlib import Path

from habitatsignal.database import ObservationStore
from habitatsignal.species import WILDLIFE_CLASS_IDS, label_for, model_class_ids


class SpeciesTests(unittest.TestCase):
    def test_only_supported_wildlife_labels_are_exposed(self):
        self.assertEqual(label_for(20), "elephant")
        self.assertIsNone(label_for(2))
        self.assertEqual(model_class_ids(), sorted(WILDLIFE_CLASS_IDS))


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

