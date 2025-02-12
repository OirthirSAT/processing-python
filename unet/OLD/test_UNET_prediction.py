# import unittest
# import numpy as np
# import os
# import tensorflow as tf
# from . import UNET_prediction
# from pathlib import Path
# import cv2
# import matplotlib.pyplot as plt

# IMAGE_SHAPE = (256, 256, 3)
# IMAGE = np.random.randint(256, size=IMAGE_SHAPE, dtype=np.uint8)
# MODEL_PATH = os.path.join(Path(__file__).parent.absolute(), "unet_coastline_model.h5")

# # Define paths to save source and target images
# SOURCE_PATH = "Sentinel2_11_001-20241014T065525Z-001/Sentinel2_11_001/chunk1_20181215T183751_20181215T184316_T11SKT.TCI_RGB_site49_ID3.npz"
# TARGET_PATH = "prediction_images/prediction_image.png"


# class UnetPredictionTests(unittest.TestCase):
#     def test_image_produced(self) -> None:
#         """
#         This test is designed to catch breaking changes.
#         """
#         self.assertTrue(os.path.exists(MODEL_PATH))

#         # Save the random image to the source path
#         np.savez(SOURCE_PATH, image=IMAGE)

#         # Make prediction and save the result to the target path
#         UNET_prediction.make_prediction(MODEL_PATH, SOURCE_PATH, TARGET_PATH)


# if __name__ == "__main__":
#     unittest.main()
