import csv
import requests
import shutil
import os
import ee

## Dataset used: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1

PROJECT_NAME = "scriptminer-oirthirsat"
DATASET = "LANDSAT/LC09/C02/T1"
BANDS = ["B5", "B4", "B3", "B2"]
TILE_WIDTH = 4096 # Pixels wide a tile should be
INPUT_BIT_DEPTH = 16
OUTPUT_BIT_DEPTH = 12
RESOLUTION = 30 # Metres per pixel

def import_sites():
    sites = []
    with open("locations.csv") as file:
        file.readline()
        csv_reader = csv.DictReader(file)
        lines = []
        for row in csv_reader:
            if not row['Location Number']:
                break # All rows read
            try:
                sites.append({'region':row['Region'],'subregion':row['Subregion'],'point':[float(row['Longitude']),float(row['Latitude'])]})
            except ValueError:
                print(f"Invalid coordinate [{row['Longitude']},{row['Latitude']}] - could not convert to float.")
                exit()
    return sites


# Calculate clarity scores for each image
def clarity_check(img, geometry):
    qa = img.select('QA_PIXEL')

    cloudShadowBitMask = 1 << 4
    snowBitMask = 1 << 5
    clearBitMask = 1 << 6

    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
        .And(qa.bitwiseAnd(snowBitMask).eq(0)) \
        .And(qa.bitwiseAnd(clearBitMask).neq(0))

    total_pixel_count = img.reduceRegion(reducer=ee.Reducer.count(), geometry=geometry,
                                         maxPixels=1e13, scale=30).values(['QA_PIXEL']).get(0)
    clear_pixel_count = img.updateMask(mask).reduceRegion(reducer=ee.Reducer.count(), geometry=geometry,
                                                          maxPixels=1e13, scale=30).values(['QA_PIXEL']).get(0)

    valid_pixel_count = img.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geometry,
        maxPixels=1e13,
        scale=30
    ).values(['QA_PIXEL']).get(0)

    all_pixel_count = img.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geometry,
        maxPixels=1e13,
        scale=30
    ).values(['QA_PIXEL']).get(0)

    # Get count of all pixels including those outwith mask (using different .values)
    clarity_score = ee.Number(clear_pixel_count).divide(total_pixel_count).multiply(ee.Number(100))
    null_pixel_ratio = ee.Number(valid_pixel_count).divide(ee.Number(all_pixel_count))
    # Check ratio between total_pixel_count and including_null_pixel_count to see if there are null pixels
    clarity_score = ee.Algorithms.If(null_pixel_ratio.lt(0.99), ee.Number(0), clarity_score) # This line is probably broken
    
    return img.set({'clarity_score': clarity_score})

# @retry(tries=10, delay=1, backoff=2)
def get_result(index, site):
    print(f'[File {index}]: Initialising')
    ee.Initialize(project=f"{PROJECT_NAME}", opt_url='https://earthengine-highvolume.googleapis.com')
    print(f"[File {index}]: Downloading point {site['region']}:{site['subregion']} {site['point']}")
    img_col = ee.ImageCollection(DATASET)
    selectors = BANDS

    bbox = ee.Geometry.Point(site['point']) #.buffer(0.5 * TILE_WIDTH * RESOLUTION).bounds() # Select a bounding box
    print("BBOX", bbox.getInfo())

    imgs_filter_list = img_col.filterDate('2024-01-01', '2025-09-08') \
        .filterBounds(bbox) \
        .map(lambda img: clarity_check(img, bbox)) \
        .filter(ee.Filter.gte('clarity_score', 80)) \
        .sort('clarity_score', False)
    
    n_imgs = imgs_filter_list.size().getInfo()
    if n_imgs == 0:
        print(f"No images found for {site['subregion']} meeting clarity requirements.")
        return None
    print(f"Found {n_imgs} image(s) meeting validity requirements.")
    # Select first image, downscale from 16 bit to 12 bit
    img = imgs_filter_list.first().select(selectors).multiply((2**(OUTPUT_BIT_DEPTH-INPUT_BIT_DEPTH))).toUint16()
    
    img = img.clip(bbox.coveringGrid(img.projection(), TILE_WIDTH * RESOLUTION).geometry())

    img_id = site["subregion"]
    print(f"Exporting {img_id} to Google Drive")
    print(img.getInfo())

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=img_id.replace("/","-"),
        folder="GEE_Exports_3",
        fileNamePrefix=img_id.replace("/","-"),
        # region=img.geometry(),
        # dimensions=TILE_WIDTH,
        scale=30,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": False, "noData": 0},
    )
    return task

if __name__ == "__main__":
    print("Importing locations...")
    sites = import_sites()
    print("LOCATIONS:", sites)
    print("Authenticating...")
    ee.Authenticate()
    print("Done Authenticating.")

    print("Creating downloads folder.")
    try:
        os.mkdir('downloads')
    except FileExistsError:
        print("Folder already exists, skipping.")
    
    print("Beginning downloads.")
    i = 0
    ids = []
    while i < len(sites):
        task = get_result(i, sites[i])
        task.start()
        print(task.id)
        ids.append(task.id)
        i += 1
    print("All downloads started.")
    print("Task IDs:", ids)

# Get task from id
for id in ids:
    task = ee.batch.Task(id, 'EXPORT_IMAGE', "READY")
    print(task.status())
