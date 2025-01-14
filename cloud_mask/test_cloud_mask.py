import unittest
import numpy as np
from cloud_mask import CloudMask


_IMAGE_SHAPE = (4, 512, 512)
_IMAGE_WITHOUT_CLOUDS = np.full(shape=_IMAGE_SHAPE, fill_value=25, dtype=np.uint8)
_IMAGE_WITH_CLOUDS = np.array(
    [
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=250),
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=250),
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=250),
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=128),
    ],
    dtype=np.uint8,
)
_RANDOM_IMAGE = np.random.uniform(low=25, high=250, size=_IMAGE_SHAPE).astype(np.uint8)
_IMAGE_WITHOUT_CLOUDS_EXPECTED_OUTPUT = np.full(
    shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=False
)
_IMAGE_WITH_CLOUDS_EXPECTED_OUTPUT = np.full(
    shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=True
)


class CloudMaskTests(unittest.TestCase):
    def test_image_without_clouds_produces_empty_mask(self) -> None:
        cloud_mask = CloudMask(bands=_IMAGE_WITHOUT_CLOUDS).create_cloud_mask()
        np.testing.assert_array_equal(cloud_mask, _IMAGE_WITHOUT_CLOUDS_EXPECTED_OUTPUT)

    def test_image_entirely_clouds_produces_full_mask(self) -> None:
        cloud_mask = CloudMask(bands=_IMAGE_WITH_CLOUDS).create_cloud_mask()
        np.testing.assert_array_equal(cloud_mask, _IMAGE_WITH_CLOUDS_EXPECTED_OUTPUT)


if __name__ == "__main__":
    unittest.main()
