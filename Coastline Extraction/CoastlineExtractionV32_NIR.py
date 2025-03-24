import os
import glob
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize(arr):
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    return (arr - arr_min) / (arr_max - arr_min) if (arr_max - arr_min) != 0 else arr

def apply_coastline_overlay(image, coastline, color=(255, 0, 255)):
    overlay = image.copy()
    overlay[coastline == 255] = color
    return overlay

def process_binary(binary_img, min_area=1000):
    inv = cv2.bitwise_not(binary_img)
    contours, hierarchy = cv2.findContours(inv.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                cv2.drawContours(binary_img, [cnt], -1, 255, thickness=cv2.FILLED)
    edges = cv2.Canny(binary_img, 50, 150)
    cnts, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coastline = np.zeros_like(binary_img)
    cv2.drawContours(coastline, cnts, -1, 255, thickness=2)
    coastline = cv2.bitwise_or(coastline, edges)
    return coastline

def process_algo1(bin_img):
    return process_binary(bin_img.copy())

def process_algo2(bin_img):
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    return process_binary(morph)

def process_algo3(bin_img):
    blur = cv2.medianBlur(bin_img, 5)
    return process_binary(blur)

def process_algo4(bin_img):
    sobel = cv2.Sobel(bin_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel = np.uint8(np.absolute(sobel))
    ret, sobel_bin = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(bin_img, sobel_bin)
    return process_binary(combined)

folder = r"image folder path"
file_pattern = os.path.join(folder, "*.tiff")
tiff_files = sorted(glob.glob(file_pattern))
if len(tiff_files) == 0:
    raise Exception("No TIFF files found in the specified folder!")
fixed_thresholds = [6, 2, 22, 19, 10, 6, 22, 15, 18, 30, 2, 6, 26, 18, 26, 26, 6, 18, 30, 10, 14, 6, 2, 10, 30, 26, 22, 6, 30]

for i, filepath in enumerate(tiff_files):
    threshold_value = fixed_thresholds[i % len(fixed_thresholds)]
    try:
        with rasterio.open(filepath) as src:
            nir = src.read(1)
            red = src.read(2)
            green = src.read(3)
            blue = src.read(4)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        continue

    nir_norm = normalize(nir)
    nir_u8 = (nir_norm * 255).astype(np.uint8)
    composite = np.dstack((nir_u8, nir_u8, nir_u8))
    ret, binary_nir = cv2.threshold(nir_u8, threshold_value, 255, cv2.THRESH_BINARY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    clahe_img = clahe.apply(nir_u8)
    ret, binary_clahe = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    combined_binary1 = cv2.bitwise_or(binary_nir, binary_clahe)
    weighted = cv2.addWeighted(binary_nir, 0.5, binary_clahe, 0.5, 0)
    
    ret, combined_binary2 = cv2.threshold(weighted, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, binary_triangle = cv2.threshold(nir_u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    binary_list = [("NIR", binary_nir), ("CLAHE", binary_clahe), ("Combined1", combined_binary1), ("Combined2", combined_binary2), ("Triangle", binary_triangle)]
    algo_funcs = [("Algo1", process_algo1), ("Algo2", process_algo2), ("Algo3", process_algo3), ("Algo4", process_algo4)]
    
    fig, axs = plt.subplots(4, 5, figsize=(25, 20))
    for r, (algo_name, func) in enumerate(algo_funcs):
        for c, (name, bin_img) in enumerate(binary_list):
            coastline = func(bin_img)
            overlay = apply_coastline_overlay(composite, coastline, color=(255, 0, 255))
            axs[r, c].imshow(overlay)
            axs[r, c].set_title(f"{algo_name} - {name}")
            axs[r, c].axis('off')
    plt.tight_layout()
    base_filename = os.path.splitext(filepath)[0]
    out_path = f"{base_filename}_4x5_overlay.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved overlay for image '{os.path.basename(filepath)}' using threshold {threshold_value}: {out_path}")
