import csv
import requests
import shutil
import os
import ee

## Dataset used: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1

PROJECT_NAME = "scriptminer-oirthirsat"
DATASET = "LANDSAT/LC09/C02/T1"
BANDS = ["B5", "B4", "B3", "B2"]
TILE_WIDTH = 1024 # Pixels wide a tile should be
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
                                         maxPixels=1e13, scale=30)
    clear_pixel_count = img.updateMask(mask).reduceRegion(reducer=ee.Reducer.count(), geometry=geometry,
                                                          maxPixels=1e13, scale=30)

    clarity_score = ee.Number(clear_pixel_count.values(['QA_PIXEL']).get(0)) \
        .divide(total_pixel_count.values(['QA_PIXEL']).get(0)).multiply(ee.Number(100))

    return img.set({'clarity_score': clarity_score})


# @retry(tries=10, delay=1, backoff=2)
def get_result(index, site):
    print(f'[File {index}]: Initialising')
    ee.Initialize(project=f"{PROJECT_NAME}", opt_url='https://earthengine-highvolume.googleapis.com')
    print(f"[File {index}]: Downloading point {site['region']}:{site['subregion']} {site['point']}")
    img_col = ee.ImageCollection(DATASET)
    selectors = BANDS

    bbox = ee.Geometry.Point(site['point']).buffer(0.5 * TILE_WIDTH * RESOLUTION).bounds() # Select a bounding box
    imgs_filter_list = img_col.filterDate('2024-01-01', '2025-01-01') \
        .filterBounds(bbox) \
        .map(lambda img: clarity_check(img, bbox)) \
        .filter(ee.Filter.gte('clarity_score', 80)) \
        .reduceColumns(ee.Reducer.toList(), ['system:index']).get('list')

    # os.mkdir(os.path.join("downloads",site['subregion']))

    for imgID in imgs_filter_list.getInfo():
        print(f"Fetching image {imgID}")
        img = ee.Image('LANDSAT/LC09/C02/T1/' + imgID).select(selectors).multiply(0.0000275).add(-0.2)

        url = img.getDownloadURL({
            "bands": selectors,
            "region": bbox,
            "scale": 30,
            "filePerBand": False,
            "format": "GEO_TIFF"
        })

        # handle downloading the actual pixels
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()
        filename = f"{site['subregion']}_{imgID}.tiff"
        
        with open(os.path.join('downloads', filename), 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print(f'[File {index}]: Download complete.')
        
        print("BREAK; taking only first file")
        break

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
    while i < len(sites):
        get_result(i, sites[i])
        i += 1