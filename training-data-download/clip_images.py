from osgeo import gdal
import os
import re

TILE_WIDTH = 4096 # Width of tile to put output images on
BIT_DEPTH = 16 # Bit depth of output
SOURCE_FOLDER_PATH = "downloads"
DESTINATION_FOLDER_PATH = "cropped"
IN_MIN, IN_MAX = 0, 65565 # Input minimum and maximum values
OUT_MIN, OUT_MAX = 0, 2**BIT_DEPTH # Output minimum and maximum values

print("Creating cropped images folder.")
try:
    os.mkdir(DESTINATION_FOLDER_PATH)
except FileExistsError:
    print("Folder already exists, skipping.")

images_to_crop = os.listdir(SOURCE_FOLDER_PATH)

for image in images_to_crop:
    if not image.endswith(".tif"):
        print(f"Skipping non-tiff file '{image}'")
        continue
    print(f"Processing '{image}'")
    raw_image_path = os.path.join(SOURCE_FOLDER_PATH, image)
    processed_image_path = os.path.join(DESTINATION_FOLDER_PATH, image)

    stats = gdal.Info(raw_image_path, stats=True)
    valid_percentage = float(re.search("STATISTICS_VALID_PERCENT=([\d.]*)", stats).group(1))

    if valid_percentage < 50:
        print(f"Only {valid_percentage}% of '{raw_image_path}' is valid. Skipping image.")
        continue

    gdal.Translate(processed_image_path, raw_image_path,
        srcWin=[0,0,TILE_WIDTH,TILE_WIDTH], # Crop to tile size
        format="GTiff", # Leave format as GeoTiff
        creationOptions=["COMPRESS=DEFLATE"], # Compress image
        bandList=[2,3,4,1], # Reorder bands to R,G,B,NIR
        scaleParams=[[IN_MIN, IN_MAX, OUT_MIN, OUT_MAX]], # Scale input from -1 to 1 float to BIT_DEPTH unsigned integer
        outputType=gdal.GDT_UInt16,
        noData=0,
        options=["-colorinterp","red,green,blue"],
        )
