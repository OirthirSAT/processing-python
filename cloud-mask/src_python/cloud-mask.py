import numpy as np
from numpy.typing import NDArray
from typing import cast

# import matplotlib.pyplot as plt


class CloudMask:
    def __init__(
        self,
        bands: NDArray[np.float64],
        ndsi_threshold: float = 0.3,
        brightness_threshold: float = 0.5,
        thermal_threshold: float = 0.5,  # example threshold
    ) -> None:
        """
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
        self.__band1 = bands[0]
        self.__band2 = bands[1]
        self.__band3 = bands[2]
        self.__band4 = bands[3]
        self.ndsi_threshold = ndsi_threshold
        self.brightness_threshold = brightness_threshold
        self.thermal_threshold = thermal_threshold

    def __compute_ndsi(self) -> NDArray[np.float64]:
        return cast(
            NDArray[np.float64],
            (self.__band1 - self.__band2) / (self.__band1 + self.__band2),
        )

    def __compute_brightness(self) -> NDArray[np.float64]:
        return cast(
            NDArray[np.float64],
            (self.__band1 + self.__band2 + self.__band3 + self.__band4) / 3,
        )

    def create_cloud_mask(self) -> NDArray[np.bool_]:
        """
        Returns:
            An MxN boolean array with the same dimension as any band of the
            input image, where a value of True indicates that the
            corresponding pixel in the input image is part of a cloud.
        """
        ndsi = self.__compute_ndsi()
        brightness = self.__compute_brightness()
        return cast(
            NDArray[np.bool_],
            (ndsi > self.ndsi_threshold)
            & (brightness > self.brightness_threshold)
            & (self.__band3 > self.thermal_threshold),
        )
