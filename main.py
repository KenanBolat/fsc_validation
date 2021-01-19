import os
import datetime
import glob
import rasterio
from rasterio.merge import merge
from collections import Counter


import numpy as np
from rasterio.plot import show
# import gdal

def convert_day_to_datetime(totalday):
    year = int(totalday[0:4])
    days = int(totalday[4:8])
    return (datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)).strftime("%Y%m%d")


start = datetime.datetime.now()

processing_path = r"/media/knn/DATA/modis_hdf_data/v05"
merge_path = r"/media/knn/DATA/modis_hdf_data/merged_daily"

tiles = [os.path.join(processing_path,f_) for f_ in glob.glob1(processing_path, "*v0*.tif")]
files_v2 = [f_.split(".")[1] for f_ in glob.glob1(processing_path, "*v02*.tif")]
files_v3 = [f_.split(".")[1] for f_ in glob.glob1(processing_path, "*v03*.tif")]
files_v4 = [f_.split(".")[1] for f_ in glob.glob1(processing_path, "*v04*.tif")]
files_v5 = [f_.split(".")[1] for f_ in glob.glob1(processing_path, "*v05*.tif")]

if Counter(files_v2).keys() == Counter(files_v3).keys() == Counter(files_v4).keys() == Counter(files_v5).keys():
    for tile in Counter(files_v2).keys():
        intermediate_date_start = datetime.datetime.now()
        date_ = convert_day_to_datetime(tile[1:])
        print(date_)
        tile_merged = [rasterio.open(f_) for f_ in tiles if f_.find(tile) > 0]
        mosaic, out_trans = merge(tile_merged)
        # show(mosaic, cmap='terrain')
        out_meta = tile_merged[0].meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "crs": "+proj=longlat +datum=WGS84 +no_defs "
                         })
        with rasterio.open(os.path.join(merge_path, "MOD10A1_" + date_ + "_europe.tif"), "w", **out_meta) as destination:
            destination.write(mosaic)

        intermediate_date_end = datetime.datetime.now()
        print(date_, "took : ", intermediate_date_end - intermediate_date_start)
        break


end = datetime.datetime.now()
duration = end - start

print("Start time:", start, "Duration:", duration)

