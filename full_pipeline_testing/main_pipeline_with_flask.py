# flask_dashboard.py
import os
from flask import Flask, render_template, send_from_directory
import matplotlib.pyplot as plt
import tifffile
from modified_cloud_mask import CloudMask
from modified_coastline_extraction import CoastlineExtractor
import numpy as np

app = Flask(__name__)

CACHE_DIR = "flask_image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def normalise_band(band):
    band = band.astype('float32')
    return (band - band.min()) / (band.max() - band.min() + 1e-6)

def save_for_flask(image, name):
    path = os.path.join(CACHE_DIR, name)
    if image.ndim == 2:  # grayscale
        plt.imsave(path, image, cmap='gray')
    else:  # RGB
        plt.imsave(path, image)
    return name

def run_pipeline(image_path):
    # Load image
    image = tifffile.imread(image_path)
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.shape[-1] <= 4:
        image = np.moveaxis(image, -1, 0)

    metadata = {"dimensions": image.shape, "image_path": image_path}

    # Cloud masking
    masker = CloudMask(bands=image)
    mask = masker.create_cloud_mask()
    masked_image = masker.apply_cloud_mask(mask)

    # Save cloud mask images
    mask_file = save_for_flask(mask.astype(np.uint8) * 255, "cloud_mask.png")
    masked_file = save_for_flask(np.nan_to_num(masked_image[..., :3]), "masked_image.png")
    metadata["mask_coverage"] = float(mask.sum()) / mask.size

    # Coastline extraction
    extractor = CoastlineExtractor()
    results = extractor.run(masked_image, method="otsu")

    ndwi_file = save_for_flask(results["ndwi_image"], "ndwi.png")
    binary_file = save_for_flask(results["binary_mask"], "binary_mask.png")
    overlay_file = save_for_flask(results["overlay_image"], "overlay.png")

    return metadata, {
        "Cloud Mask": mask_file,
        "Masked Image": masked_file,
        "NDWI": ndwi_file,
        "Binary Mask": binary_file,
        "Overlay": overlay_file
    }

# flask stuff and running, will need debug but i dont have proper files
@app.route('/')
def index():
    image_path = "raw_landsat_images/Dundee_LC09_204021_20240322.tiff"
    metadata, images = run_pipeline(image_path)

    html_sections = ""
    for label, fname in images.items():
        html_sections += f"""
        <h2>{label}</h2>
        <img src="/cache/{fname}" style="max-width:600px;"><br>
        """

    html_sections += "<h2>Metadata</h2>"
    for k, v in metadata.items():
        html_sections += f"<b>{k}:</b> {v}<br>"

    return f"""
    <html>
    <head><title>Flask Image & Data Dashboard</title></head>
    <body>
    <h1>Flask Image & Data Dashboard</h1>
    {html_sections}
    </body>
    </html>
    """

@app.route('/cache/<path:filename>')
def serve_cache(filename):
    return send_from_directory(CACHE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
