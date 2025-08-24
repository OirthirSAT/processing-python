import numpy as np
import cv2
from skimage import morphology

class CoastlineExtractor:
    """
    Assumes [0] NIR, [1] Blue, [2] Green, [3] Red
    NOTE: OpenCV uses BGR order by default, so check band indexing
    depending on how the image was loaded.
    """

    def __init__(self, min_area=500):
        self.min_area = min_area

    def normalise(self, arr):
        arr = arr.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        return (arr - arr_min) / (arr_max - arr_min + 1e-6)

    def compute_ndwi(self, image):
        # Green = band 2, NIR = band 0
        green = self.normalise(image[2])
        nir = self.normalise(image[0])
        ndwi = (green - nir) / (green + nir + 1e-6)
        return self.normalise(ndwi)

    def segment_water(self, ndwi, method="otsu"):
        ndwi_u8 = (ndwi * 255).astype(np.uint8)

        if method == "otsu":
            # Otsu thresholding
            _, mask = cv2.threshold(ndwi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == "kmeans":
            # K-means clustering with k=2
            Z = ndwi_u8.reshape((-1, 1)).astype(np.float32)
            _, labels, _ = cv2.kmeans(Z, 2, None,
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                      10, cv2.KMEANS_RANDOM_CENTERS)
            mask = labels.reshape(ndwi_u8.shape).astype(np.uint8) * 255

        elif method == "gmm":
            # Gaussian Mixture Model with 2 components using OpenCV ml.EM
            Z = ndwi_u8.reshape(-1, 1).astype(np.float32)
            em = cv2.ml.EM_create()
            em.setClustersNumber(2)
            em.trainEM(Z)
            _, labels = em.predict(Z)
            mask = labels.reshape(ndwi_u8.shape).astype(np.uint8) * 255

        else:
            raise ValueError(f"Unknown method {method}")

        # Morphological cleanup: prefer contiguous water bodies
        mask_clean = morphology.remove_small_holes(mask.astype(bool), area_threshold=500)
        mask_clean = morphology.remove_small_objects(mask_clean, min_size=self.min_area)
        mask_clean = (mask_clean.astype(np.uint8)) * 255

        return mask_clean

    def extract_boundaries(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coastline = np.zeros_like(mask)
        cv2.drawContours(coastline, contours, -1, 255, 2)
        return contours, coastline

    def overlay_boundary(self, image, boundary, color=(255, 0, 255)):
        # Use RGB preview (R,G,B)
        R, G, B = self.normalise(image[3]), self.normalise(image[2]), self.normalise(image[1])
        rgb = np.stack([R, G, B], axis=-1)
        overlay = (rgb * 255).astype(np.uint8).copy()
        overlay[boundary == 255] = color
        return overlay

    def run(self, image, mask, image_path=None, method="otsu"):
        """
        Run coastline extraction on cloud-masked image.
        Args:
            image: cloud-masked image bands [NIR, B, G, R]
            mask: cloud mask used (binary)
            image_path: optional, for metadata reporting
            method: 'otsu', 'kmeans', 'gmm'
            i need to sleep
        """
        ndwi = self.compute_ndwi(image)
        water_mask = self.segment_water(ndwi, method=method)
        contours, boundary = self.extract_boundaries(water_mask)
        overlay = self.overlay_boundary(image, boundary)

        # Prints metadata, obviously
        metadata = {
            "mask_coverage": float(mask.sum()) / mask.size,
            "dimensions": image.shape,
            "image_path": image_path
        }
        print("[INFO] Metadata:")
        for k, v in metadata.items():
            print(f"   {k}: {v}")

        return {
            "ndwi_image": ndwi,
            "binary_mask": water_mask,
            "overlay_image": overlay,
            "vector_boundary": contours
        }
