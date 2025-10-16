import unittest
import numpy as np
from cloud_mask import CloudMask

BIT_DEPTH = 12 # Assumed bit depth of input imagery

_IMAGE_SHAPE = (4, 512, 512)
_IMAGE_WITHOUT_CLOUDS = np.array(
    [
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=0.1*2**BIT_DEPTH),
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=0.1*2**BIT_DEPTH),
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=0.1*2**BIT_DEPTH),
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=0.00*2**BIT_DEPTH),
    ]
)
_IMAGE_WITH_CLOUDS = np.array(
    [
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=0.5*2**BIT_DEPTH),
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=0.5*2**BIT_DEPTH),
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=0.5*2**BIT_DEPTH),
        np.full(shape=(_IMAGE_SHAPE[1], _IMAGE_SHAPE[2]), fill_value=0.7*2**BIT_DEPTH),
    ]
)
_RANDOM_IMAGE = np.random.uniform(low=0.1*2**BIT_DEPTH, high=0.99*2**BIT_DEPTH, size=_IMAGE_SHAPE).astype(np.uint16)
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
