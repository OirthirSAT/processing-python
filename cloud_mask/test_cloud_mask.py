import unittest
import numpy as np
from cloud_mask import CloudMask


IMAGE_SHAPE = (4, 512, 512)
IMAGE_WITHOUT_CLOUDS = np.full(shape=IMAGE_SHAPE, fill_value=25, dtype=np.uint8)
IMAGE_WITH_CLOUDS = np.array(
    [
        np.full(shape=(IMAGE_SHAPE[1], IMAGE_SHAPE[2]), fill_value=250),
        np.full(shape=(IMAGE_SHAPE[1], IMAGE_SHAPE[2]), fill_value=250),
        np.full(shape=(IMAGE_SHAPE[1], IMAGE_SHAPE[2]), fill_value=250),
        np.full(shape=(IMAGE_SHAPE[1], IMAGE_SHAPE[2]), fill_value=128),
    ],
    dtype=np.uint8,
)
RANDOM_IMAGE = np.random.uniform(low=25, high=250, size=IMAGE_SHAPE).astype(np.uint8)
IMAGE_WITHOUT_CLOUDS_EXPECTED_OUTPUT = np.full(
    shape=(IMAGE_SHAPE[1], IMAGE_SHAPE[2]), fill_value=False
)
IMAGE_WITH_CLOUDS_EXPECTED_OUTPUT = np.full(
    shape=(IMAGE_SHAPE[1], IMAGE_SHAPE[2]), fill_value=True
)


class CloudMaskTests(unittest.TestCase):
    def test_image_without_clouds_produces_empty_mask(self) -> None:
        cloud_mask = CloudMask(bands=IMAGE_WITHOUT_CLOUDS).create_cloud_mask()
        np.testing.assert_array_equal(cloud_mask, IMAGE_WITHOUT_CLOUDS_EXPECTED_OUTPUT)

    def test_image_entirely_clouds_produces_full_mask(self) -> None:
        cloud_mask = CloudMask(bands=IMAGE_WITH_CLOUDS).create_cloud_mask()
        np.testing.assert_array_equal(cloud_mask, IMAGE_WITH_CLOUDS_EXPECTED_OUTPUT)


if __name__ == "__main__":
    unittest.main()
