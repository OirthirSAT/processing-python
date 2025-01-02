# This script demonstrates how to preprocess an image from an .npz file and make a prediction using a trained model.
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Define a function to preprocess an image from an .npz file
def preprocess_npz(npz_path):

    # Load the .npz file
    data = np.load(npz_path)

    # grab the 'image' key which holds the relevant image data
    image = data["image"]

    # Resize the image to 256x256
    image_resized = cv2.resize(image, (256, 256))

    # Normalize the image to [0, 1]
    image_normalized = image_resized.astype("float32") / 255.0

    # Ensure the image has 3 channels (RGB) (not greyscale)
    if image_normalized.shape[-1] != 3:
        raise ValueError(
            f"Image must have 3 channels (RGB). Got shape: {image_normalized.shape}"
        )

    image_swapped = image_normalized[
        ..., [2, 1, 0]
    ]  # Swap the first (Red) and third (Blue) channels

    # Expand dimensions to match the input shape (batch_size, height, width, channels) of the model
    image_batch = np.expand_dims(image_swapped, axis=0)

    return image_batch


"""
# Example usage:
npz_path = "input/chunk1_20181215T183751_20181215T184316_T11SKT.TCI_RGB_site49_ID3.npz" # Path to the .npz file
image_batch = preprocess_npz(npz_path) # Preprocess the image

# Load the model from the model_maker.py script trained on the traineddata folder
model = tf.keras.models.load_model('unet_coastline_model.h5')

# Make prediction
prediction = model.predict(image_batch)

# Display the prediction using matplotlib
# If the model produces a segmentation mask, display it accordingly
plt.imshow(prediction[0])  # Display the first image in the batch
plt.show()
"""


def make_prediction(model_path: str, source_path: str, target_path: str):
    """
    Parameters:
        model_path: Path to the saved model, e.g. "unet_coastline_model.h5".
        source_path: Path to the source image as an npz.
        target_path: Path to save prediction to. Must end with a valid file extension e.g. "prediction.png".
    """
    image_batch = preprocess_npz(source_path)
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(image_batch)
    # prediction_image.show()
    plt.imshow(prediction[0])
    plt.savefig(target_path)
