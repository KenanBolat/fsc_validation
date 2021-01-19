import os
import numpy as np
import datetime
import glob
import gdal
import rasterio
from rasterio.merge import merge
from rasterio.plot import show

def convert_day_to_datetime(totalday):
    year = int(totalday[0:4])
    days = int(totalday[4:8])
    return datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)


start = datetime.datetime.now()

processing_path = r"/media/knn/DATA/modis_hdf_data/tif"

from collections import Counter
files_v2 = [f_.split(".")[1] for f_ in glob.glob1(processing_path, "*v02*.tif")]
files_v3 = [f_.split(".")[1] for f_ in glob.glob1(processing_path, "*v03*.tif")]
files_v4 = [f_.split(".")[1] for f_ in glob.glob1(processing_path, "*v04*.tif")]
files_v5 = [f_.split(".")[1] for f_ in glob.glob1(processing_path, "*v05*.tif")]

if Counter(files_v2).keys() == Counter(files_v3).keys() == Counter(files_v4).keys() == Counter(files_v5):
    pass
















end = datetime.datetime.now()
duration = end - start

print("Start time:", start, "Duration:", duration)

