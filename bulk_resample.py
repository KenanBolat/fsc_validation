from osgeo import gdal
from osgeo import gdalconst
import os
import glob
import datetime

processing_path = "/media/knn/DATA/modis_hdf_data/merged_daily"
reference_file = os.path.join(processing_path, "..", "mars_daily", "h35_20181101_day_TSMS.tif")


def resample_merged_data(filename):
    reference = gdal.Open(reference_file, 0)  # this opens the file in only reading mode
    referenceTrans = reference.GetGeoTransform()
    x_res = referenceTrans[1]
    y_res = -referenceTrans[5]  # make sure this value is positive

    # specify input and output filenames
    inputFile = "Path to input file"
    outputFile = "Path to output file"

    # call gdal Warp
    kwargs = {"format": "GTiff", "xRes": x_res, "yRes": y_res}
    ds = gdal.Warp(outputFile, inputFile, **kwargs)

    del output


def list_merged_data(path_, pattern):
    return [os.path.join(path_, row) for row in glob.glob1(path_, pattern)]
