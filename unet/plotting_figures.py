import cv2  # type: ignore[assignment]
import os
import numpy as np
import tensorflow as tf  # type: ignore
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from pathlib import Path
from typing import Any, Tuple
from numpy.lib.npyio import NpzFile
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from unet_prediction import ImageProcessor


# Load the model
model = tf.keras.models.load_model("unet_coastline_model.h5")
input_directory = "raw_data/Sentinel2_11_001"
output_directory = "prediction_images/sentinel_predictions_24trainedimages"
os.makedirs(output_directory, exist_ok=True)
class_labels = ["Land", "Water", "No Data"]
colors = ["blue", "green", "red"]


for file_name in os.listdir(input_directory):
    if file_name.endswith(".npz"):  # Ensure it's an npz file
        npz_path = os.path.join(input_directory, file_name)

        try:
            # Preprocess the image
            processed_image = ImageProcessor.preprocess_npz(npz_path)
            print(f"Processed {file_name}, shape: {processed_image.shape}")

            # Load original image for visualization
            original_image = np.load(npz_path)["image"]
            original_image_resized = cv2.resize(original_image, (256, 256))

            # Make prediction
            prediction = model.predict(processed_image)

            # Display the original image and prediction side by side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(original_image_resized)
            axes[0].set_title(f"Original - {file_name}")
            axes[0].axis("off")

            axes[1].imshow(prediction[0], cmap=plt.colormaps["tab10"])
            axes[1].set_title("Prediction")
            axes[1].axis("off")

            legend_elements = [
                mpatches.Patch(color="green", label="Land"),
                mpatches.Patch(color="red", label="Water"),
                mpatches.Patch(color="blue", label="No Data"),
            ]
            plt.legend(handles=legend_elements, loc="upper left", fontsize=10)

            # Save the prediction image
            plt.savefig(f"{output_directory}/{file_name}.png")

            plt.close()  # Close the figure to prevent overlapping plots

        except ValueError as e:
            print(f"Skipping {file_name} due to error: {e}")
