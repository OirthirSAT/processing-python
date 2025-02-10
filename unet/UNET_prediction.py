
# This script demonstrates how to preprocess an image from an .npz file and make a prediction using a trained model.
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from pathlib import Path
from typing import Any, Tuple


class ImageProcessor:
    @staticmethod
    def preprocess_npz(npz_path: str) -> NDArray[np.floating[Any]]:
        """
        Args:
            npz_path: Path to the .npz file containing the rgb image as a numpy array under
                key "image".

        Returns:
            A 4-dimensional array containing [the image as bgr], as the model accepts an
            additional dimension representing batches of images.

        Raises:
            ValueError: If the input npz does not represent an RGB image with shape
                (x, y, 3).
        """
        # Load the .npz file
        data: NDArray[np.floating[Any]] = np.load(npz_path)


        # grab the 'image' key which holds the relevant image data
        image: NDArray[np.floating[Any]] = data["image"]


        # Resize the image to 256x256
        image_resized: NDArray[np.floating[Any]] = cv2.resize(image, (256, 256))


        # Normalize the image to [0, 1]
        image_normalized: NDArray[np.floating[Any]] = image_resized.astype("float32") / 255.0


        # Ensure the image has 3 channels (RGB) (not greyscale)
        if image_normalized.shape[-1] != 3:
            raise ValueError(
                f"Image must have 3 channels (RGB). Got shape: {image_normalized.shape}"
            )

        image_swapped: NDArray[np.floating[Any]] = image_normalized[
            ..., [2, 1, 0]
        ]  # Swap the first (Red) and third (Blue) channels

        # Expand dimensions to match the input shape (batch_size, height, width, channels) of the model
        image_batch: NDArray[np.floating[Any]] = np.expand_dims(image_swapped, axis=0)

        return image_batch


class ModelPredictor:
    @staticmethod
    def make_prediction(model_path: str, source_path: str, target_path: str) -> None:
        """
        Runs the specified network on a source image, and saves its output to the target
        path.

        Args:
            model_path: Path to the saved model, e.g. "unet_coastline_model.h5".
            source_path: Path to the source image as an npz.
            target_path: Path to save prediction to. Must end with a valid file extension e.g. "prediction.png".
        """
        image_batch: NDArray[np.floating[Any]] = ImageProcessor.preprocess_npz(source_path)
        model: tf.keras.Model = tf.keras.models.load_model(model_path)
        prediction: NDArray[np.floating[Any]] = model.predict(image_batch)
        plt.imshow(prediction[0])
        plt.savefig(target_path)




# Example usage
model_path: str = "unet_coastline_model.h5"
source_path: str = "RAW_DATA/Landsat8_11_001/duck_2014-01-04-15-36-14_L8_rgbr_ID2.npz"
target_path: str = "prediction_images/target.png"


ModelPredictor.make_prediction(model_path, source_path, target_path)
