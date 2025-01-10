import cv2
import numpy as np
import tensorflow as tf  # type: ignore
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Any, cast


# typedefs
_NUMERIC_ARRAY = NDArray[np.floating[Any] | np.integer[Any]]


def preprocess_npz(
    npz_path: str | None = None,
    np_image: _NUMERIC_ARRAY | None = None,
) -> NDArray[np.floating[Any]]:
    """
    Args:
        npz_path: Path to the .npz file containing the rgb image as a numpy array under
            key "image", or none if image data is directly provided.
        np_image: Numeric array representing the image to run the prediction on, in
            RGB.

    Returns:
        A 4-dimensional array containing [the image as bgr], as the model accepts an
        additional dimension representing batches of images.

    Raises:
        ValueError: If the input does not represent an RGB image with shape
            (x, y, 3), or if both npz_image and np_image are none.
    """

    if npz_path is None and np_image is None:
        raise ValueError("Expected one of npz_image or np_image to be not None.")

    if np_image is None:
        npz_path = cast(str, npz_path)  # MyPy needs some help

        data = np.load(npz_path)

        # grab the 'image' key which holds the relevant image data
        np_image = data["image"]

    np_image = cast(_NUMERIC_ARRAY, np_image)  # MyPy needs some help

    # Unet admits images of size (256, 256) only.
    image_resized = cv2.resize(np_image, (256, 256))

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

    # Model admits shape (batch_size, height, width, channels) only.
    image_batch = np.expand_dims(image_swapped, axis=0)

    return image_batch


def make_prediction(
    model_path: str,
    source_path: str | None = None,
    source_image: _NUMERIC_ARRAY | None = None,
    target_path: str | None = None,
) -> NDArray[np.floating[Any]]:
    """
    Runs the specified network on a source image, and saves its output to the target
    path.

    Args:
        model_path: Path to the saved model, e.g. "unet_coastline_model.h5".
        source_path: Path to the source image in rgb as an npz. Must be provided if
            source_image is None.
        source_image: Raw source image in rgb as an array. If not None, source_path
            will be disregarded.
        target_path: Path to save prediction to. Must end with a valid file extension
            e.g. "prediction.png".

    Returns:
        The prediction of the model, as a 4d array with the first dimension
            representing batches, in BGR with values between [0, 1)
    """

    image_batch = preprocess_npz(source_path, source_image)
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(image_batch)
    if target_path:
        plt.imshow(prediction[0])
        plt.savefig(target_path)
    prediction = cast(NDArray[np.floating[Any]], prediction)  # keras has no MyPy stubs
    return prediction
