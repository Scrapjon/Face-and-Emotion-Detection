import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_emotion.anti_spoofing.train_liveness import make_datasets


class TestTrainLivenessDatasets(unittest.TestCase):
    def _write_dummy_image(self, path: Path) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(path)

    def test_make_datasets_returns_copy_safe_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            for split in ("train", "val"):
                for label in ("live", "spoof"):
                    (root / split / label).mkdir(parents=True, exist_ok=True)
                    samples = 2 if split == "train" else 1
                    for index in range(samples):
                        self._write_dummy_image(root / split / label / f"{index}.jpg")

            train_ds, val_ds = make_datasets(root)

            train_images, train_labels = train_ds[0]
            val_images, val_labels = val_ds[0]

            self.assertEqual(train_images.shape, (4, 224, 224, 3))
            self.assertEqual(train_labels.shape, (4,))
            self.assertEqual(val_images.shape, (2, 224, 224, 3))
            self.assertEqual(val_labels.shape, (2,))
            self.assertEqual(set(train_labels.tolist()), {0.0, 1.0})
            self.assertEqual(set(val_labels.tolist()), {0.0, 1.0})

            self.assertIsNotNone(copy.deepcopy(train_ds))
            self.assertIsNotNone(copy.deepcopy(val_ds))


if __name__ == "__main__":
    unittest.main()
