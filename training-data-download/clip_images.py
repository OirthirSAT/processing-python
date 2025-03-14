from osgeo import gdal
import os
import re

TILE_WIDTH = 1024
BIT_DEPTH = 16
SOURCE_FOLDER_PATH = "downloads"
DESTINATION_FOLDER_PATH = "cropped"

print("Creating cropped images folder.")
try:
    os.mkdir(DESTINATION_FOLDER_PATH)
except FileExistsError:
    print("Folder already exists, skipping.")

images_to_crop = os.listdir(SOURCE_FOLDER_PATH)

for image in images_to_crop:
    raw_image_path = os.path.join(SOURCE_FOLDER_PATH, image)
    processed_image_path = os.path.join(DESTINATION_FOLDER_PATH, image)

    stats = gdal.Info(raw_image_path, stats=True)
    valid_percentage = float(re.search("STATISTICS_VALID_PERCENT=([\d.]*)", stats).group(1))

    if valid_percentage < 100:
        print(f"Only {valid_percentage}% of '{raw_image_path}' is valid. Skipping image.")
        continue

    gdal.Translate(".tmp.tiff", raw_image_path,
                   srcWin=[0,0,TILE_WIDTH,TILE_WIDTH], # Crop to tile size
                   format="GTiff", # Leave format as GeoTiff
                   creationOptions=["COMPRESS=DEFLATE"], # Compress image
                   bandList=[2,3,4,1], # Reorder bands to R,G,B,NIR
                   scaleParams=[[-1, 1,0,2**(BIT_DEPTH)]], # Scale input from -1 to 1 float to BIT_DEPTH unsigned integer
                   outputType=gdal.GDT_UInt16,
                   noData=0,
                   )
    gdal.Translate(processed_image_path, ".tmp.tiff", options=["-colorinterp","red,green,blue,alpha"]) # Add colour interpretation information to bands
    os.remove(".tmp.tiff")