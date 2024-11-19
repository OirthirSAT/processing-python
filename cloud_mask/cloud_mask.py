import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Any, cast, Tuple
import warnings


class CloudMask:
    def __init__(
        self,
        bands: NDArray[np.uint8] | None = None,
        tif_path: str = None,
        downsample_factor: float = 0.1,
        ndsi_threshold: float = 0.3,
        brightness_threshold: float = 0.5,
        thermal_threshold: float = 0.5,  # example threshold
    ) -> None:
        """
        Initialise using 4-band NDArray (default) or .tif file.

        Parameters:
            bands: The four bands of the image, as a 4xMxN tensor. Having the
                input as an array ensures M and N are consistent for each band.
            nsdi_threshold: Any pixel with an nsdi value above this threshold
                will be masked out.
            brightness_threshold: Any pixel with a brightness value above this
                threshold will be masked out.
            thermal_threshold: Any pixel with a thermal value above this
                threshold will be masked out.
        """

        self.downsample_factor = downsample_factor
        self.ndsi_threshold = ndsi_threshold
        self.brightness_threshold = brightness_threshold
        self.thermal_threshold = thermal_threshold

        if bands is not None:
            self._band_red = bands[0]
            self._band_green = bands[1]
            self._band_blue = bands[2]
            self._band_nir = bands[3]
        elif tif_path is not None:
            (
                self._band_red,
                self._band_green,
                self._band_blue,
                self._band_nir,
            ) = self._readfile(tif_path, self.downsample_factor)
        else:
            raise ValueError(
                "Cannot initialise CloudMask: no bands data or tif file path specified."
            )

    def _readfile(
        self, file: str, downsample_factor: float
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
        """Load image bands from BGR image file"""
        image_bgr: NDArray[np.uint8] = cast(np.uint8, cv2.imread(file))

        if image_bgr is None:
            raise FileNotFoundError(
                f"Could not load file '{file}'. Check file exists and is of '.tif' format."
            )

        # Compressing file for speed
        new_size: Tuple[int, int] = (
            int(image_bgr.shape[0] * downsample_factor),
            int(image_bgr.shape[1] * downsample_factor),
        )

        image_resized: NDArray[np.uint8] = cast(
            NDArray[np.uint8],
            cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA),
        )

        band_red: NDArray[np.uint8] = image_resized[:, :, 2]
        band_green: NDArray[np.uint8] = image_resized[:, :, 1]
        band_blue: NDArray[np.uint8] = image_resized[:, :, 0]
        band_nir: NDArray[np.uint8] = np.zeros_like(band_red)
        warnings.warn(
            "Importing NIR channel from .tif not implemented. Setting band to 0s. This could cause unexpected masking results."
        )

        return band_red, band_green, band_blue, band_nir

    def _compute_ndsi(self) -> NDArray[np.floating[Any]]:
        """Compute Normalised Difference Snow Index (NDSI).

        Compute the NDSI for each pixel using the green and near infrared
        bands using the formula:
        (Green - NIR) / (Green + NIR)
        Returns:
            An MxN float array with each pixel representing the computed
            NDSI.
        """
        return cast(
            NDArray[np.floating[Any]],
            (self._band_green.astype(float) - self._band_nir.astype(float))
            / (self._band_green.astype(float) + self._band_nir.astype(float)),
        )

    def _compute_brightness(self) -> NDArray[np.floating[Any]]:
        """Compute brightnesses array.

        Returns:
            An MxN float array with each pixel representing the average
            brightness across all 4 channels.
        """
        return cast(
            NDArray[np.floating[Any]],
            (
                self._band_red.astype(float)
                + self._band_green.astype(float)
                + self._band_blue.astype(float)
                + self._band_nir.astype(float)
            )
            / 4
            / 256,
        )

    def create_cloud_mask(self) -> NDArray[np.bool_]:
        """Create a boolean mask of clouded areas.

        Returns:
            An MxN boolean array with the same dimension as any band of the
            input image, where a value of True indicates that the
            corresponding pixel in the input image is part of a cloud.
        """
        ndsi: NDArray[np.floating[Any]] = self._compute_ndsi()
        brightness: NDArray[np.floating[Any]] = self._compute_brightness()
        return cast(
            NDArray[np.bool_],
            (ndsi >= self.ndsi_threshold)
            & (brightness >= self.brightness_threshold)
            & (self._band_nir >= self.thermal_threshold),
        )

    def apply_cloud_mask(
        self, cloud_mask: NDArray[np.bool_]
    ) -> NDArray[np.floating[Any]]:
        """Combine bands and apply cloud mask."""
        bands = [self._band_red, self._band_green, self._band_blue, self._band_nir]

        masked_image = np.array(
            np.dstack(bands), dtype=float
        )  # Shape: (rows, columns, bands). Convert to float to allow NaN values.
        masked_image[cloud_mask, :] = (
            np.nan
        )  # Set pixels where cloud_mask is False to NaN
        return masked_image

    @staticmethod
    def visualise_image(image: NDArray[np.uint8]) -> None:
        plt.figure(figsize=(10, 10))
        plt.imshow(
            image.astype(np.uint8)[:, :, :3]
        )  # Only visually render RGB channels
        plt.show()

    @staticmethod
    def cloud_cover_fraction(cloud_mask: NDArray[np.bool_]) -> float:
        """Compute the fraction of cloud_mask is covered in cloud.

        Take an existing cloud mask and compute what fraction is cloud. It is
        assumed clouded pixels are True, non-clouded pixels are False.
        """
        return cast(
            float,
            cloud_mask.sum() / cloud_mask.size
        )


# Loading image bands and creating cloud mask, then visualizing results

if __name__ == "__main__":
    file: str = "../marching_squares/Aberdeenshire.tif"

    # Aberdenshire.tif does not have a NIR channel. This is automatically initialised to 0, so thresholds
    # need adjusted from sensible values to make test work.
    cloud_masker = CloudMask(
        tif_path=file, downsample_factor=0.1, ndsi_threshold=1.0, thermal_threshold=0.0
    )

    cloud_mask: NDArray[np.bool_] = cloud_masker.create_cloud_mask()
    image: NDArray[np.floating[Any]] = cloud_masker.apply_cloud_mask(
        cloud_mask=cloud_mask
    )
    print(f"Cloud Cover: {CloudMask.cloud_cover_fraction(cloud_mask)*100:.3f}%")
    CloudMask.visualise_image(image)
