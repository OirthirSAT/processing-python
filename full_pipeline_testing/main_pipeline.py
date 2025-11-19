import os
import json
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from cloud_mask.cloud_mask import CloudMask

#NEW marching squares and segmentation algorithm
from full_pipeline_testing.modified_coastline_extraction import CoastlineExtractor
from full_pipeline_testing.marching_squares_refined import MarchingSquaresRefiner

#ORIGINAL marching squares and segmentation algorithm
from full_pipeline_testing.modified_marching_squares_altseg import CoastlineExtractor_MS_altseg

#Initiliasing file paths
image_path = "../test-images/Aberdyfi.tif"#"raw_landsat_images/Dundee_LC09_204021_20240322.tiff"
output_dir = "output"
initialisation_dir = os.path.join(output_dir, "initialisation")
cloudmask_dir = os.path.join(output_dir, "cloud_mask")
data_output_dir = os.path.join(output_dir, "data_output")

#check the directories exist
os.makedirs(initialisation_dir, exist_ok=True)
os.makedirs(cloudmask_dir, exist_ok=True)
os.makedirs(data_output_dir, exist_ok=True)




def normalise_band(band):
    #convert band to a float32 object and normalise
    band = band.astype("float32")
    return (band - band.min()) / (band.max() - band.min() + 1e-6)

def load_image(image_path):
    
    image = tifffile.imread(image_path) #using the tiffile package from open-cv
    
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    
    elif image.shape[-1] <= 4:
        image = np.moveaxis(image, -1, 0) 
    
    print(f"[INFO] Loaded image: {image.shape}")
    return image

#prints min, max and mean values for each band rgb and nir 
def show_band_stats(image):
    
    for i in range(image.shape[0]):
        
        band = image[i , : , :]
        print(f"[STATS] Band {i}: min={band.min()}, max={band.max()}, mean={band.mean():.2f}")
        
        norm = normalise_band(band)
        plt.imsave(f"{initialisation_dir}/band_{i}.png", norm, cmap="gray")

#saves the image to a compatible image format for viewing 
def save_rgb_preview(image):
    
    if image.shape[0] < 4:
        raise ValueError("Expected at least 4 bands for RGB+NIR")
    
    #Assuming the bands are in order [NIR,BLUE,GREEN,RED]
    R, G, B = image[0], image[1], image[2]
    rgb = np.stack([R, G, B], axis=-1)
    
    dynamic_range = [250,600]
    rgb_norm = (rgb.clip(dynamic_range[0], dynamic_range[1]).astype(float)-dynamic_range[0])/(dynamic_range[1]-dynamic_range[0])
    
    plt.imsave(f"{initialisation_dir}/rgb_preview.png", rgb_norm)
    
    print("[INFO] RGB preview saved")


def run_cloud_masking(image, image_path):
    
    print("[INFO] Running cloud masking...")
    #Defining an object of class CloudMask
    
    masker = CloudMask(
        bands=image,
        downsample_factor=0.1,
        ndsi_threshold=0.5  ,
        brightness_threshold=0.3,
        thermal_threshold=0.0,
    )
    
    #Create the mask
    low_res_mask = masker.create_cloud_mask()

    #resizes the low resolution mask to the original image size
    mask_full = resize(
        low_res_mask.astype(float), image.shape[1:], order=0, preserve_range=True
    ) > 0.5

    #Saves the mask
    mask_path = os.path.join(cloudmask_dir, "cloud_mask.tiff")
    tifffile.imwrite(mask_path, mask_full.astype(np.uint8) * 255)
    print(f"[INFO] Saved cloud mask to {mask_path}")

    #Applies the mask to our image
    masked = masker.apply_cloud_mask(mask_full)

    #Saves the masked image
    masked_path = os.path.join(cloudmask_dir, "masked_image.tiff")
    tifffile.imwrite(masked_path, masked.astype(np.float32), dtype=np.float32)
    print(f"[INFO] Saved masked image to {masked_path}")

    CloudMask.visualise_image(masked)

    return mask_full, masked


def pipeline(image_path, use_altseg=False):

    image = load_image(image_path)
    show_band_stats(image)
    save_rgb_preview(image)

    mask, masked_image = run_cloud_masking(image, image_path)

    metadata = {
        "mask_coverage": float(mask.sum()) / mask.size,
        "dimensions": image.shape,
        "image_path": image_path,
    }

    print("[INFO] Running coastline extraction...")
    
    if use_altseg:
        # Using original marching squares class
        print("[INFO] Using original coastline extraction")
        extractor = CoastlineExtractor_MS_altseg()
        # Runs the extraction code.
        results = extractor.run(masked_image, 0.1)
        
        # Note: results dictionary keys should match what the altseg class returns
        # If necessary, modify the keys below to match class
        preprocessed_image = results["preprocessed_image"]
        CloudMask.visualise_image(preprocessed_image)
        binary_mask = results["threshold_image"] #black and white segmented image
        overlay = results["overlay_image"] #image with coastline outlined 
        vector_boundary = results["shapes"] #the coastline segment vectors, shapes[0] should be the coastline vector outlined.

    else:
        extractor = CoastlineExtractor()
        results = extractor.run(masked_image, method="otsu")
        binary_mask = results["binary_mask"]
        overlay = results["overlay_image"]
        vector_boundary = results["vector_boundary"]

    # Save the images: binary cloud mask and the image with outlined coastline
    #plt.imsave(f"{data_output_dir}/ndwi.png", results.get("ndwi_image", binary_mask), cmap="gray")
    plt.imsave(f"{data_output_dir}/binary_mask.png", binary_mask, cmap="gray")
    plt.imsave(f"{data_output_dir}/overlay.png", overlay)
    
    print(f"[INFO] Extracted {len(vector_boundary)} coastline contours")

    metadata["coastline_contours"] = len(vector_boundary)

    # print("[INFO] Refining coastline with marching squares...")
    # refiner = MarchingSquaresRefiner()
    # refined_results = refiner.run(binary_mask)

    # # Save refined vectors (as JSON for now)
    # refined_path = os.path.join(data_output_dir, "refined_vectors.json")
    
    # with open(refined_path, "w") as f:
    #     json.dump([list(map(list, vec)) for vec in refined_results["vectors"]], f, indent=2)
    
    # print(f"[INFO] Saved refined coastline vectors to {refined_path}")

    # metadata["refined_vectors"] = len(refined_results["vectors"])

    json_path = os.path.join(data_output_dir, "metadata.json")
    
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[INFO] Saved metadata JSON to {json_path}")


if __name__ == "__main__":
    # Set use_altseg=True to use the Original Marching Squares and segmentation algorithm
    pipeline(image_path, use_altseg=True)
