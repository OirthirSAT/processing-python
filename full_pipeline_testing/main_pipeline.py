import datetime
import os
import json
import logging
import cv2
import gc
import timeit
import numpy as np
import matplotlib.pyplot as plt
# from skimage.transform import resize

from cloud_mask.cloud_mask import CloudMask
from chain_encode.chain_encode import ChainEncode

#NEW marching squares and segmentation algorithm
# from full_pipeline_testing.modified_coastline_extraction import CoastlineExtractor
# from full_pipeline_testing.marching_squares_refined import MarchingSquaresRefiner

#ORIGINAL marching squares and segmentation algorithm
from full_pipeline_testing.modified_marching_squares_altseg import CoastlineExtractor_MS_altseg

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

#Initiliasing file paths
images_folder = "../test-images"
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
    
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) #using the tiffile package from open-cv
    
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    
    elif image.shape[-1] <= 4:
        image = np.moveaxis(image, -1, 0) 
    
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
        ndsi_threshold=0.2  ,
        brightness_threshold=0.12,
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

t_estimate = 141/98286

def timed_pipeline(image_path, name, out_dir, ndwi_bias=0.00, cm_ndsi=0.4, cm_bright=0.18, cm_therm=0.2, limit=300_000):
    """Cut-down version of pipeline with extra logging and outputs removed."""
    timings = {}
    global t_estimate
    timings["_ndwi_bias"] = ndwi_bias
    timings["_downsample_factor"] = 1
    timings["_cm_ndsi_threshold"] = cm_ndsi
    timings["_cm_brightness_threshold"] = cm_bright
    timings["_cm_thermal_threshold"] = cm_therm
    s = timeit.default_timer()
    image = load_image(image_path)
    timings["load_image"] = timeit.default_timer() - s

    s = timeit.default_timer()
    masker = CloudMask(
        bands=image,
        downsample_factor=1,
        ndsi_threshold=cm_ndsi,
        brightness_threshold=cm_bright,
        thermal_threshold=cm_therm
    )
    
    # Cloud Mask
    print("CM", end="..", flush=True)
    mask = masker.create_cloud_mask()
    timings["cloud_mask_create"] = timeit.default_timer() - s

    s = timeit.default_timer()
    masked_image = masker.apply_cloud_mask(mask)
    timings["cloud_mask_apply"] = timeit.default_timer() - s
    del image
    gc.collect()

    plt.imsave(os.path.join(out_dir, f"{name}_binary_mask.png"), mask, cmap="gray")
    del mask
    gc.collect()
    # Marching Squares
    print("OTSU", end="..", flush=True)
    s = timeit.default_timer()
    extractor = CoastlineExtractor_MS_altseg()
    preprocessed_image, valid_mask = CoastlineExtractor_MS_altseg._preprocess_image(masked_image, 1)
    timings["marchingsquares_preprocess"] = timeit.default_timer() - s

    s = timeit.default_timer()
    _, threshold_image, threshold_value = CoastlineExtractor_MS_altseg._otsu_segmentation_4channel(preprocessed_image, valid_mask, ndwi_bias=ndwi_bias)
    timings["marchingsquares_otsu_segmentation"] = timeit.default_timer() - s
    del preprocessed_image
    del valid_mask, threshold_value
    gc.collect()

    plt.imsave(os.path.join(out_dir, f"{name}_threshold_image.png"), threshold_image, cmap="gray")

    print("List vectors", end="..", flush=True)
    s = timeit.default_timer()
    state_array, x_len, y_len = CoastlineExtractor_MS_altseg._point_array(threshold_image)
    vectors = CoastlineExtractor_MS_altseg._list_vectors(state_array, x_len, y_len)
    timings["marchingsquares_list_vectors"] = timeit.default_timer() - s
    del state_array, x_len, y_len
    del threshold_image
    gc.collect()

    print(f"{len(vectors)} ({t_estimate*len(vectors)})", end="..", flush=True)
    if len(vectors) > limit:
        print("Too many vectors, abandoning.")
        return len(vectors)

    print("Vector shapes", end="..", flush=True)
    s = timeit.default_timer()
    shapes = CoastlineExtractor_MS_altseg._vector_shapes(vectors)
    timings["marchingsquares_vector_shapes"] = timeit.default_timer() - s
    t_estimate = timings["marchingsquares_vector_shapes"] / len(vectors)
    del vectors
    gc.collect()

    # Chain Encoding
    print("Chain encode", end="..", flush=True)
    s = timeit.default_timer()
    ChainEncode.chain_encode(shapes, os.path.join(out_dir, f"{name}_boundaries.npz"))
    timings["chain_encode"] = timeit.default_timer() - s
    del shapes
    gc.collect()

    with open(os.path.join(out_dir, f"{name}_timings.json"), "w") as f:
        json.dump(timings, f)
    
    # return timings

def pipeline(image_path, use_altseg=False):

    image = load_image(image_path)
    print(f"[INFO] Loaded image: {image.shape}")
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
        results = extractor.run(masked_image, 1)
        
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

    print("[INFO] Encoding, compressing and saving coastline vectors")
    ChainEncode.chain_encode(vector_boundary, f"{data_output_dir}/encoded_boundaries.npz")

    print("[INFO] Saving output images")
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
    # pipeline(image_path, use_altseg=True)

    location_vec_lens = [['Torrisdale.tif', '324514'], ['Rothesay.tif', '1340688'], ['River Dee.tif', '1580553'], ['Plymouth.tif', '808635'], ['North Skye.tif', '846323'], ['Norfolk.tif', '513428'], ['Margate.tif', '493659'], ['Magilligan Point.tif', '350840'], ['Kildonan, Arran.tif', '757742'], ['Gwynedd.tif', '730759'], ['Dundee.tif', '624548'], ['Dumbarton.tif', '1339483'], ['Culbin.tif', '1344705'], ['Carlisle.tif', '489310'], ['Barrow-in-Furness.tif', '404664'], ['Balivanich, South Uist.tif', '846323'], ['Aberdyfi.tif', '730759'], ['Winchelsea.tif', '98286'], ['Tredrissi.tif', '111213'], ['Towan Beach, Cornwall.tif', '22876'], ['Torrisdale.tif', '324514'], ['Tiree.tif', '204000'], ['Stromness.tif', '76305'], ['Stoke, Kent.tif', '208146'], ['St Ishmael.tif', '145639'], ['Rothesay.tif', '1340688'], ['Rockcliffe.tif', '107812']]

    for iter in range(100):
        output_dir = f"timings_full_extraction_fixed_{iter}"
        os.makedirs(output_dir, exist_ok=True)
        failed = []
        # for img in ['Magilligan Point.tif', 'Barrow-in-Furness.tif', 'Carlisle.tif', 'Margate.tif', 'Norfolk.tif', 'Dundee.tif', 'Gwynedd.tif', 'Aberdyfi.tif', 'Kildonan, Arran.tif', 'Plymouth.tif', 'North Skye.tif', 'Balivanich, South Uist.tif', 'Dumbarton.tif', 'Rothesay.tif', 'Culbin.tif', 'River Dee.tif']:
        # for img in os.listdir(images_folder)[::-1]:
        print(list(map(lambda x: x, sorted(location_vec_lens, key=lambda loc:int(loc[1])))))
        for img in map(lambda x: x[0], sorted(location_vec_lens, key=lambda loc:int(loc[1]))):
            if not img.endswith(".tif"):
                continue
            # if img[:5] > "Porta":
            #     continue
            print(f"[{datetime.datetime.now()}] Processing {img} ({output_dir})... ", end="", flush=True)
            s = timeit.default_timer()
            err = timed_pipeline(os.path.join(images_folder, img), img.split(".")[0], output_dir, limit=1e9)
            if err:
                failed.append(img)
            print(f"{timeit.default_timer() - s:.3f}s")
        
        # Compute all images which failed because they were too large in the first pass
        for img in failed:
            print(f"[{datetime.datetime.now()}] Processing {img} ({output_dir})... ", end="", flush=True)
            s = timeit.default_timer()
            err = timed_pipeline(os.path.join(images_folder, img), img.split(".")[0], output_dir, limit=1e9)
            print(f"{timeit.default_timer() - s:.3f}s")
        
