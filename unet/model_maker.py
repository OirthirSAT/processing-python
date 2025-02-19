import os
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
import cv2  # type: ignore[assignment]
from typing import Any

# Define paths
npz_folder: str = "trained_data"
output_model_path: str = "unet_coastline_model.h5"

# Step 1: Load data from `.npz` files
input_array: list[NDArray[Any]] = []
label_array: list[NDArray[Any]] = []

# Track skipped files for debugging
skipped_files: list[str] = []
mismatch_files: list[str] = []

# Iterate over all .npz files
for file in os.listdir(npz_folder):
    if file.endswith(".npz"):
        data: dict[str, NDArray[Any]] = np.load(os.path.join(npz_folder, file))
        print(f"Loading {file}...")

        # Check the available keys in the current file
        print(f"Available keys in {file}: {data.keys()}")

        # Load input image
        if "image" in data:
            input_image: NDArray[Any] = data["image"]
        else:
            print(f"Skipping {file}, 'image' key not found.")
            continue

        # Load label
        if "label" in data:
            label: NDArray[Any] = data["label"]
        else:
            print(f"Skipping {file}, 'label' key not found.")
            continue

        print(f"Input shape: {input_image.shape}")
        print(f"Label shape: {label.shape}")

        # Resize images and labels to 256x256
        target_size: tuple[int, int] = (256, 256)
        input_resized: NDArray[Any] = cv2.resize(input_image, target_size)
        label_resized: NDArray[Any] = cv2.resize(label, target_size)

        # Ensure the label has the correct number of channels
        if label_resized.ndim == 2:
            label_resized = np.expand_dims(label_resized, axis=-1)

        if label_resized.shape[-1] == 1:
            label_resized = tf.keras.utils.to_categorical(label_resized, num_classes=3)

        if label_resized.shape[-1] != 3:
            print(
                f"Warning: Label shape {label_resized.shape} does not have 3 channels. Skipping {file}."
            )
            skipped_files.append(file)
            continue

        input_array.append(input_resized)
        label_array.append(label_resized)

# Check skipped files
print(f"Skipped {len(skipped_files)} files due to label shape issues.")
print(f"Skipped files: {skipped_files}")

# Ensure input_data and label_data match
if len(input_array) != len(label_array):
    print(
        f"Mismatch in number of samples: input={len(input_array)}, label={len(label_array)}."
    )
    mismatch_files = [
        f
        for f in os.listdir(npz_folder)
        if f.endswith(".npz") and f not in skipped_files
    ]
    print(f"Files causing mismatch: {mismatch_files}")
else:
    print(f"Loaded {len(input_array)} samples successfully.")

# Convert to NumPy arrays
if len(input_array) == len(label_array):
    input_data: NDArray[np.float32] = np.array(input_array, dtype=np.float32) / 255.0
    label_data: NDArray[np.int32] = np.array(label_array, dtype=np.int32)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        input_data, label_data, test_size=0.2, random_state=42
    )

    # Step 4: Define the U-Net model
    def unet_model(input_shape: tuple[int, int, int]) -> tf.keras.Model:
        inputs = layers.Input(input_shape)
        c1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
        c1 = layers.MaxPooling2D((2, 2))(c1)
        c2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c1)
        c2 = layers.MaxPooling2D((2, 2))(c2)
        c3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c2)
        u2 = layers.UpSampling2D((2, 2))(c3)
        u2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u2)
        u1 = layers.UpSampling2D((2, 2))(u2)
        outputs = layers.Conv2D(3, (1, 1), activation="softmax")(u1)
        return models.Model(inputs, outputs)

    # Compile and train model
    input_shape: tuple[int, int, int] = X_train.shape[1:]
    model: tf.keras.Model = unet_model(input_shape)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16
    )

    # Save model
    model.save(output_model_path)
    print(f"Model saved to {output_model_path}")
else:
    print("Data mismatch detected. Model training aborted.")
