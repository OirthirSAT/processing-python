import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

# Function to apply coastline overlay with adjustable color
def apply_coastline_overlay(image, coastline, color=(255, 0, 0)):
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay[coastline == 255] = color 
    return overlay


#image_path = Path("Aberdeenshire_S2_20220810_TCI.tif")
image_path = Path("UK30m.tiff")
image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

# Handle images with an alpha channel
if image.shape[2] == 4:  # BGRA
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # Convert to BGR

# Convert to HSV for better color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower_blue1 = np.array([90, 50, 50])
upper_blue1 = np.array([130, 255, 255])
lower_blue2 = np.array([80, 60, 50])  # Adjusted for better range
upper_blue2 = np.array([120, 255, 255])  # Adjusted for more blue/green range


# Create masks for both color ranges
mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)

# Combine masks
filler_mask = cv2.bitwise_or(mask1, mask2)

# Invert filler mask to focus on actual map area
map_mask = cv2.bitwise_not(filler_mask)
masked_image = cv2.bitwise_and(image, image, mask=map_mask)

# Reshape the image for K-means clustering
Z = masked_image.reshape((-1, 3))
Z = np.float32(Z)

# K-means clustering to segment land and water
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 3  # Land, water, and potential transitional zones
_, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back to 8-bit and reshape to original image shape
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()].reshape((image.shape))

# Identify the water cluster (assuming it's the darkest cluster)
gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
water_mask = np.where(gray_segmented == gray_segmented.min(), 255, 0).astype(np.uint8)

# Morphological operations to clean up small artifacts
kernel = np.ones((3, 3), np.uint8)  

cleaned_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=3)

# Edge detection to refine boundaries
edges = cv2.Canny(cleaned_mask, 50, 150)

# Combine contours with edges for a refined coastline
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

coastline = np.zeros_like(gray_segmented)
cv2.drawContours(coastline, contours, -1, (255), thickness=2)
coastline = cv2.bitwise_or(coastline, edges)

# Remove fully enclosed clusters
contours, _ = cv2.findContours(coastline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) < 40000:  # Adjust area threshold as needed
        cv2.drawContours(coastline, [cnt], -1, (0), thickness=cv2.FILLED)

# Overlay coastline on images
average_image = cv2.addWeighted(masked_image, 0.5, segmented_image, 0.5, 0)
average_with_coastline = apply_coastline_overlay(average_image, coastline, color=(255, 0, 0))
original_with_coastline = apply_coastline_overlay(image, coastline, color=(255, 0, 0))


cv2.imwrite("average_with_coastline.png", cv2.cvtColor(average_with_coastline, cv2.COLOR_RGB2BGR))
cv2.imwrite("original_with_coastline.png", cv2.cvtColor(original_with_coastline, cv2.COLOR_RGB2BGR))

plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)
plt.title('Masked Image')
plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title('Segmented (K-means)')
plt.imshow(segmented_image)

plt.subplot(2, 2, 3)
plt.title('Average With Coastline')
plt.imshow(original_with_coastline, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Extracted Coastline')
plt.imshow(coastline, cmap='gray')

plt.tight_layout()
plt.show()