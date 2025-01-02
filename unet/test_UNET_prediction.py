import tempfile
import unittest
import numpy as np
import os
import UNET_prediction
from pathlib import Path

IMAGE_SHAPE = (256, 256, 3)
IMAGE = np.random.randint(256, size=IMAGE_SHAPE, dtype=np.uint8)
MODEL_PATH = os.path.join(
    Path(__file__).parent.absolute(), "UNET/unet_coastline_model.h5"
)


class UnetPredictionTests(unittest.TestCase):
    def test_image_produced(self):
        """
        This test is designed to catch breaking changes.
        """
        self.assertTrue(os.path.exists(MODEL_PATH))
        _, source = tempfile.mkstemp(suffix=".npz")
        _, target = tempfile.mkstemp(suffix=".png")
        np.savez(source, image=IMAGE)
        UNET_prediction.make_prediction(MODEL_PATH, source, target)


if __name__ == "__main__":
    unittest.main()
