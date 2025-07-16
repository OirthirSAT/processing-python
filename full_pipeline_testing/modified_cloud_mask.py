import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Any, cast, Tuple
import tifffile
import warnings


class CloudMask:
    print("[INFO] running Cloud Mask")

    """
    Initialise using 4-band NDArray (default) or .tif file.

    Args:
        bands: The four bands of the image, as a 4xMxN tensor. Having the input as an
            array ensures M and N are consistent for each band.
        tif_path: Path to the input file as a .tif. Only applied if bands is None.
        downsample_factor: float by which to scale the image on each axis. e.g. 0.5
            applied to a 1024x1024 image results in a 512x512 image for a 4x reduction
            in pixels.
        ndsi_threshold: Any pixel with an nsdi value above this threshold will be
            masked out.
        brightness_threshold: Any pixel with a brightness value above this threshold
            will be masked out.
        thermal_threshold: Any pixel with a thermal value above this threshold will be
            masked out.

    Raises:
        ValueError: if both tif_file and bands are None.
    """

    def __init__(
        self,
        bands: NDArray[np.uint8] | None = None,
        tif_path: str | None = None,
        downsample_factor: float = 0.1,
        ndsi_threshold: float = 0.3,
        brightness_threshold: float = 0.5,
        thermal_threshold: float = 0.5,
    ) -> None:

        self.downsample_factor = downsample_factor
        self.ndsi_threshold = ndsi_threshold
        self.brightness_threshold = brightness_threshold
        self.thermal_threshold = thermal_threshold

        if bands is not None:
            self._band_red = bands[0]
            self._band_green = bands[1]
            self._band_blue = bands[2]
            self._band_nir = bands[3]

            self._band_red_full = self._band_red
            self._band_green_full = self._band_green
            self._band_blue_full = self._band_blue
            self._band_nir_full = self._band_nir
        elif tif_path is not None:
            (
                self._band_red,
                self._band_green,
                self._band_blue,
                self._band_nir,
            ) = self._readfile(tif_path)
        else:
            raise ValueError(
                "Cannot initialise CloudMask: no bands data or tif file path specified."
            )

    def _readfile(
        self, file: str
    ) -> Tuple[
        NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]
    ]:
        """
        Load image bands from TIFF image file.

        Args:
            file: The path to a .tif image file.
            downsample_factor: A float by which to scale the image on each axis. e.g.
                0.5 applied to a 1024x1024 image results in a 512x512 image for a 4x
                reduction in pixels.

        Returns:
            The four bands of the image in a tuple, (red, green, blue, nir).

        Raises:
            FileNotFoundError: If file cannot be read.
            ValueError: If the TIFF does not contain at least 3 bands.
        """

        try:
            full_res_image: NDArray[np.uint8] = cast(
                NDArray[np.uint8], tifffile.imread(file)
            )
        except Exception:
            raise FileNotFoundError(
                f"Could not load file '{file}'. Check file exists and is of '.tif' format."
            )

        if full_res_image.ndim != 3 or full_res_image.shape[0] < 3:
            raise ValueError("TIFF file must have at least 3 bands (channels).")

        if full_res_image.shape[0] == 4:
            band_red_full = full_res_image[0, :, :]
            band_green_full = full_res_image[1, :, :]
            band_blue_full = full_res_image[2, :, :]
            band_nir_full = full_res_image[3, :, :]
        elif full_res_image.shape[0] == 3:
            band_red_full = full_res_image[0, :, :]
            band_green_full = full_res_image[1, :, :]
            band_blue_full = full_res_image[2, :, :]
            band_nir_full = np.zeros_like(band_red_full)
            warnings.warn(
                "NIR band missing in TIFF file. Band will be initialised to zeros."
            )
        else:
            raise ValueError(
                f"Unexpected number of bands ({full_res_image.shape[0]}) in TIFF file."
            )

        # Downsample bands for masking
        new_size: Tuple[int, int] = (
            int(band_red_full.shape[1] * self.downsample_factor),
            int(band_red_full.shape[0] * self.downsample_factor),
        )

        self._band_red_full = band_red_full
        self._band_green_full = band_green_full
        self._band_blue_full = band_blue_full
        self._band_nir_full = band_nir_full

        band_red = cast(NDArray[np.uint8], cv2.resize(band_red_full, new_size, interpolation=cv2.INTER_AREA))
        band_green = cast(NDArray[np.uint8], cv2.resize(band_green_full, new_size, interpolation=cv2.INTER_AREA))
        band_blue = cast(NDArray[np.uint8], cv2.resize(band_blue_full, new_size, interpolation=cv2.INTER_AREA))
        band_nir = cast(NDArray[np.uint8], cv2.resize(band_nir_full, new_size, interpolation=cv2.INTER_AREA))

        return band_red, band_green, band_blue, band_nir

    def _compute_ndsi(self) -> NDArray[np.floating[Any]]:
        green = self._band_green.astype(float)
        nir = self._band_nir.astype(float)
        denominator = green + nir
        denominator[denominator == 0] = 1e-6
        return cast(NDArray[np.floating[Any]], (green - nir) / denominator)

    def _compute_brightness(self) -> NDArray[np.floating[Any]]:
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
        ndsi = self._compute_ndsi()
        brightness = self._compute_brightness()
        return cast(
            NDArray[np.bool_],
            (ndsi >= self.ndsi_threshold)
            & (brightness >= self.brightness_threshold)
            & (self._band_nir.astype(float) / 255 >= self.thermal_threshold),
        )

    def apply_cloud_mask(
        self, cloud_mask: NDArray[np.bool_]
    ) -> NDArray[np.floating[Any]]:
        bands = [self._band_red_full, self._band_green_full, self._band_blue_full, self._band_nir_full]

        masked_image = np.array(
            np.dstack(bands), dtype=float
        )

        mask_resized = cast(
            NDArray[np.bool_],
            cv2.resize(cloud_mask.astype(np.uint8),
                       (masked_image.shape[1], masked_image.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        )

        masked_image[mask_resized, :] = np.nan
        return masked_image

    @staticmethod
    def visualise_image(image: NDArray[np.floating[Any]]) -> None:
        plt.figure(figsize=(10, 10))
        plt.imshow(image.astype(np.uint8)[:, :, :3])
        plt.show()

    @staticmethod
    def cloud_cover_fraction(cloud_mask: NDArray[np.bool_]) -> float:
        return cast(float, cloud_mask.sum() / cloud_mask.size)


if __name__ == "__main__":
    file: str = "../marching_squares/Aberdeenshire.tif"

    cloud_masker = CloudMask(
        tif_path=file, downsample_factor=0.1, ndsi_threshold=1.0, thermal_threshold=0.0
    )

    cloud_mask: NDArray[np.bool_] = cloud_masker.create_cloud_mask()
    image: NDArray[np.floating[Any]] = cloud_masker.apply_cloud_mask(
        cloud_mask=cloud_mask
    )
    print(f"Cloud Cover: {CloudMask.cloud_cover_fraction(cloud_mask)*100:.3f}%")
    CloudMask.visualise_image(image)
