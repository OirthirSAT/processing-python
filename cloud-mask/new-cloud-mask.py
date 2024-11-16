#Installing packages and libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import cv2
from typing import Tuple

#Function for loading image bands and reading in the image file
def readfile(file: str, downsample_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_bgr: np.ndarray = cv2.imread(file)

    # Compressing file for speed
    new_size: Tuple[int, int] = (
        int(image_bgr.shape[0] * downsample_factor),
        int(image_bgr.shape[1] * downsample_factor)
    )
    
    image_resized: np.ndarray = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
    
    band1: np.ndarray = image_resized[:, :, 0]
    band2: np.ndarray = image_resized[:, :, 1]
    band3: np.ndarray = image_resized[:, :, 2]
    band4: np.ndarray = np.zeros_like(band1)
    
    return band1, band2, band3, band4

#Function for computing NDSI (Normalized Difference Snow Index)
def compute_ndsi(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
    return (band1 - band2) / (band1 + band2)

#Function for computing brightness
def compute_brightness(band1: np.ndarray, band2: np.ndarray, band3: np.ndarray, band4: np.ndarray) -> np.ndarray:
    return (band1 + band2 + band3 + band4) / 4

#Main function to create cloud mask
def create_cloud_mask(band1: np.ndarray, band2: np.ndarray, band3: np.ndarray, band4: np.ndarray) -> np.ndarray:
    ndsi: np.ndarray = compute_ndsi(band1, band2)
    brightness: np.ndarray = compute_brightness(band1, band2, band3, band4)
    thermal_threshold: np.ndarray = band3 > 0.01  # Example threshold

    cloud_mask: np.ndarray = (ndsi > 0.01) & (brightness > 0.02) & thermal_threshold  # Combine indices
    return cloud_mask

#Function to apply cloud mask to image
def result(band1: np.ndarray, band2: np.ndarray, band3: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    where_1: Tuple[np.ndarray, np.ndarray] = np.where(cloud_mask == 0)
    
    band1[where_1] = 0
    band2[where_1] = 0
    band3[where_1] = 0

    image: np.ndarray = np.zeros((np.shape(band1)[0], np.shape(band1)[1], 3))
    for i in range(np.shape(band1)[0]):
        for j in range(np.shape(band1)[1]):
            image[i, j] = [band3[i, j], band2[i, j], band1[i, j]]
    
    return image

#Visualizing results
def visualize_results(band1: np.ndarray, band2: np.ndarray, band3: np.ndarray, band4: np.ndarray, cloud_mask: np.ndarray, image: np.ndarray) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image.astype(np.uint8))
    plt.show()

#Loading image bands and creating cloud mask, then visualizing results
file: str = "test-image.tif"

band1, band2, band3, band4 = readfile(file, 0.1)  #0.1 for speed purposes
cloud_mask: np.ndarray = create_cloud_mask(band1, band2, band3, band4)
image: np.ndarray = result(band1, band2, band3, cloud_mask)
visualize_results(band1, band2, band3, band4, cloud_mask, image)


