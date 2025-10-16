import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Any, cast, Tuple

EPSILON = 1e-6 # Small constant to avoid division by zero in NDSI calculation
BIT_DEPTH = 12 # Assumed bit depth of input imagery

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CloudMask:
    """
    Initialise using 4-band NDArray (default) or .tif file.

    Args:
        bands: The four bands of the image, as a 4xMxN tensor. Having the input as an
            array ensures M and N are consistent for each band.
        tif_path: Path to the input file as a .tif. Only applied if bands is None.
        downsample_factor: float by which to scale the image on each axis. e.g. 0.5
            applied to a 1024x1024 image results in a 512x512 image for a 4x reduction
            in pixels.
        ndsi_threshold: Any pixel with an nsdi value below this threshold will be
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
        thermal_threshold: float = 0.5,  # example threshold
    ) -> None:

        self.downsample_factor = downsample_factor
        self.ndsi_threshold = ndsi_threshold
        self.brightness_threshold = brightness_threshold
        self.thermal_threshold = thermal_threshold

        if bands is not None:
            logger.info("Initialising CloudMask with provided bands array.")
            self._band_red = bands[0]
            self._band_green = bands[1]
            self._band_blue = bands[2]
            self._band_nir = bands[3]
        elif tif_path is not None:
            logger.info(f"Initialising CloudMask with image file: '{tif_path}'")
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
    ) -> Tuple[
        NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]
    ]:
        """
        Load image bands from .tif image file.

        Args:
            file: The path to a .tif image file (3 or 4 channel).
            downsample_factor: A float by which to scale the image on each axis. e.g.
                0.5 applied to a 1024x1024 image results in a 512x512 image for a 4x
                reduction in pixels.

        Returns:
            The four bands of the image in a tuple, (red, green, blue, nir).

        Raises:
            FileNotFoundError: If file cannot be read, e.g. the path does not exist or
                the file is of an invalid type.
        """
        try:
            full_res_image: NDArray[np.uint8] = cast(NDArray[np.uint8], cv2.imread(file, cv2.IMREAD_UNCHANGED))
            logger.info(f"Loaded image '{file}' of shape: {full_res_image.shape}")
        except Exception as e:
            # TODO: More specific exception handling
            raise FileNotFoundError(
                f"Could not load file '{file}'. Check file exists and is of '.tif' format."
            ) from e
        
        # Handle downsampling
        if downsample_factor == 1:
            logger.info("Downsample factor is 1, using full resolution image.")
            image_resized = full_res_image
        elif downsample_factor > 0:
            # Determine new image size
            new_size: Tuple[int, int] = (
                int(full_res_image.shape[0] * downsample_factor),
                int(full_res_image.shape[1] * downsample_factor),
            )
            logger.info(f"Downsample factor is {downsample_factor}, resizing image from {full_res_image.shape[:2]} to {new_size}.")
            image_resized: NDArray[np.uint8] = cast(
                NDArray[np.uint8],
                cv2.resize(full_res_image, new_size, interpolation=cv2.INTER_AREA),
            )
        else:
            raise ValueError("Downsample_factor must be > 0.")

        band_red: NDArray[np.uint8] = image_resized[:, :, 2]
        band_green: NDArray[np.uint8] = image_resized[:, :, 1]
        band_blue: NDArray[np.uint8] = image_resized[:, :, 0]

        # Special handling for images without NIR band
        if full_res_image.shape[2] == 3:
            band_nir: NDArray[np.uint8] = np.zeros_like(band_red)
            logger.warn(
                "No NIR band available in '{file}'. Setting NIR band to 0s. This could cause unexpected masking results."
            )
        elif full_res_image.shape[2] == 4:
            # NIR band is present, load as normal
            band_nir = image_resized[:, :, 3]
        else:
            raise ValueError(
                f"Unexpected number of bands ({full_res_image.shape[2]}) in '{file}'."
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
        logger.info("Computing NDSI.")
        green = self._band_green.astype(float)
        nir = self._band_nir.astype(float)
        denominator = green + nir
        denominator[denominator == 0] = EPSILON  # Avoid division by zero
        return cast(NDArray[np.floating[Any]], (green - nir) / denominator)

    def _compute_brightness(self) -> NDArray[np.floating[Any]]:
        """Compute brightnesses array.

        Returns:
            An MxN float array with each pixel representing the average
            brightness across all 4 channels.
        """
        logger.info("Computing brightness.")
        return cast(
            NDArray[np.floating[Any]],
            (
                (
                    self._band_red
                    + self._band_green
                    + self._band_blue
                    + self._band_nir
                )
                / 4
            ).astype(float)
            / 2**BIT_DEPTH,  # Normalize based on bit depth (e.g., 12-bit imagery)
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
        thermal: NDArray[np.floating[Any]] = self._band_nir.astype(float) / 2**BIT_DEPTH
        logger.info("Computing NDSI and brightness statistics for debugging.")
        logger.info("NDSI stats (threshold=%s): min=%s, max=%s, mean=%s", self.ndsi_threshold, ndsi.min(), ndsi.max(), ndsi.mean())
        logger.info("Brightness stats (threshold=%s): min=%s, max=%s, mean=%s", self.brightness_threshold, brightness.min(), brightness.max(), brightness.mean())
        logger.info("Thermal band stats (threshold=%s): min=%s, max=%s, mean=%s", self.thermal_threshold, thermal.min(), thermal.max(), thermal.mean())
        return cast(
            NDArray[np.bool_],
            (ndsi <= self.ndsi_threshold)
            & (brightness >= self.brightness_threshold)
            & (thermal >= self.thermal_threshold),
        )

    def apply_cloud_mask(
        self, cloud_mask: NDArray[np.bool_]
    ) -> NDArray[np.floating[Any]]:
        """
        Combine bands and apply cloud mask.

        Args:
            cloud_mask: A boolean array with a shape matching the image height and
                width, where 'False' represents a pixel that should be masked.

        Returns:
            The masked image, with clouded areas (as specified by the cloud_mask
            arg) set to NaN.
        """
        logger.info("Applying cloud mask to image.")
        bands = [self._band_red, self._band_green, self._band_blue, self._band_nir]

        masked_image = np.array(
            np.dstack(bands), dtype=float
        )  # Shape: (rows, columns, bands). Convert to float to allow NaN values.
        masked_image[cloud_mask, :] = np.nan # Set pixels where cloud_mask is False to NaN
        return masked_image

    @staticmethod
    def visualise_image(image: NDArray[np.floating[Any]]) -> None:
        plt.figure(figsize=(10, 10))        
        dynamic_range = [250, 600] # Could calculate these dynamically with np.nanquantile rather than hardcoding
        plt.imshow((image[:, :, :3].clip(dynamic_range[0], dynamic_range[1]).astype(float)-dynamic_range[0])/(dynamic_range[1]-dynamic_range[0]))  # Only visually render RGB channels
        plt.show()

    @staticmethod
    def cloud_cover_fraction(cloud_mask: NDArray[np.bool_]) -> float:
        """
        Compute the fraction of cloud_mask is covered in cloud.

        Take an existing cloud mask and compute what fraction is cloud. It is
        assumed clouded pixels are True, non-clouded pixels are False.

        Args:
            cloud_mask: A boolean array with a shape matching the image height and
                width, where 'False' represents a pixel that should be masked.

        Returns:
            a float representing the ratio of masked image : image.
        """

        return cast(float, cloud_mask.sum() / cloud_mask.size)


# Loading image bands and creating cloud mask, then visualizing results

if __name__ == "__main__":
    file: str = "../Dundee.tif"

    # These values for thresholds are extremely rough estimates. Thermal is essentially
    # ignored, NDSI seems to have minimal influence, brightness is the main controller.
    cloud_masker = CloudMask(
        tif_path=file, downsample_factor=0.1, ndsi_threshold=0.2, thermal_threshold=0.0, brightness_threshold=0.12
    )

    cloud_mask: NDArray[np.bool_] = cloud_masker.create_cloud_mask()
    image: NDArray[np.floating[Any]] = cloud_masker.apply_cloud_mask(
        cloud_mask=cloud_mask
    )
    logger.info(f"Cloud Cover: {CloudMask.cloud_cover_fraction(cloud_mask)*100:.3f}%")
    CloudMask.visualise_image(image)
