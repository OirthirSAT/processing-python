from osgeo import gdal
import os

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
    image_path = os.path.join(SOURCE_FOLDER_PATH, image)
    gdal.Translate(os.path.join(DESTINATION_FOLDER_PATH, image), os.path.join(SOURCE_FOLDER_PATH, image),
                   srcWin=[0,0,TILE_WIDTH,TILE_WIDTH],
                   format="GTiff",
                   creationOptions=["COMPRESS=DEFLATE"],
                   scaleParams=[[-1, 1,-2**(BIT_DEPTH-1),(2**BIT_DEPTH)-1]],
                   outputType=gdal.GDT_Int16,
                   noData=0,
                   )