import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from modified_cloud_mask import CloudMask 
from modified_coastline_extraction import CoastlineExtractor

image_path = "raw_landsat_images/Dundee_LC09_204021_20240322.tiff"
output_dir = "output"
initialisation_dir = os.path.join(output_dir, "initialisation")
cloudmask_dir = os.path.join(output_dir, "cloud_mask")
data_output_dir = os.path.join(output_dir, "data_output")

os.makedirs(initialisation_dir, exist_ok=True)
os.makedirs(cloudmask_dir, exist_ok=True)
os.makedirs(data_output_dir, exist_ok=True)

def normalise_band(band):
    band = band.astype('float32')
    return (band - band.min()) / (band.max() - band.min() + 1e-6)

def load_image(image_path):
    image = tifffile.imread(image_path)
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.shape[-1] <= 4:
        image = np.moveaxis(image, -1, 0)
    print(f"[INFO] Loaded image: {image.shape}")
    return image

def show_band_stats(image):
    for i in range(image.shape[0]):
        band = image[i]
        print(f"[STATS] Band {i}: min={band.min()}, max={band.max()}, mean={band.mean():.2f}")
        norm = normalise_band(band)
        plt.imsave(f"{initialisation_dir}/band_{i}.png", norm, cmap='gray')

def save_rgb_preview(image):
    if image.shape[0] < 4:
        raise ValueError("Expected at least 4 bands for RGB+NIR")
    R, G, B = image[3], image[2], image[1]
    rgb = np.stack([R, G, B], axis=-1)
    rgb_norm = normalise_band(rgb)
    plt.imsave(f"{initialisation_dir}/rgb_preview.png", rgb_norm)
    print("[INFO] RGB preview saved")

def run_cloud_masking(image, image_path):
    print("[INFO] Running cloud masking...")
    masker = CloudMask(
        bands=image,
        downsample_factor=0.1,
        ndsi_threshold=0.3,
        brightness_threshold=0.5,
        thermal_threshold=0.5
    )
    low_res_mask = masker.create_cloud_mask()

    from skimage.transform import resize
    mask_full = resize(low_res_mask.astype(float), image.shape[1:], order=0, preserve_range=True) > 0.5
    mask_path = os.path.join(cloudmask_dir, "cloud_mask.tiff")
    tifffile.imwrite(mask_path, mask_full.astype(np.uint8) * 255)
    print(f"[INFO] Saved cloud mask to {mask_path}")

    masked = masker.apply_cloud_mask(mask_full)
    masked_path = os.path.join(cloudmask_dir, "masked_image.tiff")
    tifffile.imwrite(masked_path, masked.astype(np.float32), dtype=np.float32)
    print(f"[INFO] Saved masked image to {masked_path}")

    return mask_full

def pipeline(image_path):
    image = load_image(image_path)
    show_band_stats(image)
    save_rgb_preview(image)
    mask = run_cloud_masking(image, image_path)

    metadata = {
        "mask_coverage": float(mask.sum()) / mask.size,
        "dimensions": image.shape,
        "image_path": image_path
    }

    import json
    json_path = os.path.join(data_output_dir, "metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata JSON to {json_path}")

def pipeline(image_path):
    image = load_image(image_path)
    show_band_stats(image)
    save_rgb_preview(image)

    mask = run_cloud_masking(image, image_path)

    metadata = {
        "mask_coverage": float(mask.sum()) / mask.size,
        "dimensions": image.shape,
        "image_path": image_path
    }

    print("[INFO] Metadata:")
    for k, v in metadata.items():
        print(f"   {k}: {v}")

    masker = CloudMask(bands=image)
    masked_image = masker.apply_cloud_mask(mask)

    print("[INFO] Running coastline extraction...")
    extractor = CoastlineExtractor()
    results = extractor.run(masked_image, method="otsu")

    plt.imsave(f"{data_output_dir}/ndwi.png", results["ndwi_image"], cmap="gray")
    plt.imsave(f"{data_output_dir}/binary_mask.png", results["binary_mask"], cmap="gray")
    plt.imsave(f"{data_output_dir}/overlay.png", results["overlay_image"])

    print(f"[INFO] Extracted {len(results['vector_boundary'])} coastline contours")




pipeline(image_path)
