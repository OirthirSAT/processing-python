#modules required

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2  # For resizing images if needed


# Define paths
npz_folder = "traineddata"  # Replace with your folder path containing .npz files
output_model_path = "unet_coastline_model.h5"  # File to save trained model

# Step 1: Load data from `.npz` files
input_data = []
label_data = []

# Track skipped files for debugging
skipped_files = []
mismatch_files = []  # Track files causing data mismatch

# Iterate over all .npz files in the trainddata folder to grab the relevant keys - 'image' and 'label'
for file in os.listdir(npz_folder):
    if file.endswith(".npz"):
        data = np.load(os.path.join(npz_folder, file))
        print(f"Loading {file}...")

        # Check the available keys in the current file
        print(f"Available keys in {file}: {data.keys()}")

        # Load input image from the 'image' key
        if 'image' in data:
            input_image = data['image']
        else:
            print(f"Skipping {file}, 'image' key not found.")
            continue  # Skip if no appropriate image key exists

        # Load label from the 'label' key (adjust based on your dataset)
        if 'label' in data:
            label = data['label']
        else:
            print(f"Skipping {file}, 'label' key not found.")
            continue  # Skip if no appropriate label key exists

        # Print the shape of the image and label to check compatibility
        print(f"Input shape: {input_image.shape}")
        print(f"Label shape: {label.shape}")

        # Resize images and labels to a fixed size (256x256)
        target_size = (256, 256)
        input_data.append(cv2.resize(input_image, target_size))  # Resize the input image
        label_resized = cv2.resize(label, target_size)  # Resize label to match input size

        # Ensure the label has the correct number of channels
        if label_resized.ndim == 2:  # If label is 2D, add a dummy channel for segmentation (1 class)
            label_resized = np.expand_dims(label_resized, axis=-1)
        
        # Ensure the label has 3 channels, one for each class (adjust according to your task)
        if label_resized.shape[-1] == 1:  # Assuming the label has a single channel (e.g., class IDs)
            label_resized = tf.keras.utils.to_categorical(label_resized, num_classes=3)  # Adjust `num_classes`

        # Check if all labels have the same number of channels (e.g., 3 channels)
        if label_resized.shape[-1] != 3:
            print(f"Warning: Label shape {label_resized.shape} does not have 3 channels. Skipping this example.")
            skipped_files.append(file)  # Track the skipped files
            continue  # Skip labels with unexpected shape

        label_data.append(label_resized)

# Check how many samples were skipped
print(f"Skipped {len(skipped_files)} files due to label shape issues.")
print(f"Skipped files: {skipped_files}")

# Ensure input_data and label_data have the same length
if len(input_data) != len(label_data):
    print(f"Mismatch in number of samples: input_data has {len(input_data)} samples, label_data has {len(label_data)} samples.")
    mismatch_files = [f for f in os.listdir(npz_folder) if f.endswith(".npz") and f not in skipped_files]
    print(f"Files causing the mismatch: {mismatch_files}")
else:
    print(f"Loaded {len(input_data)} samples successfully.")

# If data lengths are consistent, proceed to train-test split
if len(input_data) == len(label_data):
    input_data = np.array(input_data)
    label_data = np.array(label_data)

    # Step 2: Preprocess data (e.g., normalize input, ensure labels are correct shape)
    input_data = input_data.astype('float32') / 255.0  # Scale to [0, 1] if working with images
    label_data = label_data.astype('int32')  # Ensure labels are integers if needed

    # Step 3: Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(input_data, label_data, test_size=0.2, random_state=42)

    # Step 4: Define the U-Net model
    def unet_model(input_shape):
        inputs = layers.Input(input_shape)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.MaxPooling2D((2, 2))(c1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c1)
        c2 = layers.MaxPooling2D((2, 2))(c2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c2)
        u2 = layers.UpSampling2D((2, 2))(c3)
        u2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
        u1 = layers.UpSampling2D((2, 2))(u2)
        outputs = layers.Conv2D(3, (1, 1), activation='softmax')(u1)  # Multi-channel output (3 channels)
        model = models.Model(inputs, outputs)
        return model

    # Step 5: Compile and train the model
    input_shape = X_train.shape[1:]  # Assume inputs are (height, width, channels)
    model = unet_model(input_shape)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16
    )

    # Step 6: Save the trained model as `.h5`
    model.save(output_model_path)

    print(f"Model saved to {output_model_path}")
else:
    print("Data mismatch detected. Model training aborted.")
