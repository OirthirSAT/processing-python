import os
import unittest
from PIL import Image
from tempfile import mkstemp
import numpy as np
from marching_squares import MarchingSquares

_IMAGE_SHAPE = (512, 512, 4)
_RANDOM_IMAGE = np.random.uniform(low=0, high=256, size=_IMAGE_SHAPE).astype(np.uint8)

class MarchingSquaresTests(unittest.TestCase):

    def test_no_runtime_errors(self) -> None:
        """
        This test does not verify the logic of the MarchingSquares classe, but verifies
        that it at least does not crash. This ensures that the code on main is runnable
        at all times.
        """
        fd, path = mkstemp()
        with os.fdopen(fd, "wb") as file:
            image = Image.fromarray(_RANDOM_IMAGE, "RGB")
            image.save(file, format="TIFF")
        MarchingSquares.run(path, 0.05)


if __name__ == "__main__":
    unittest.main()
