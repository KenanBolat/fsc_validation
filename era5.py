import gdal
import netCDF4 as nc
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import datetime
import rioxarray
# from distributed import Client
# client = Client()


## Plotting
import georaster

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import matplotlib.cm as cm

import dask.array as da


def masking(in_data, threshold):
    data_inter = []
    data = in_data[0].values
    data_inter = np.where(data <= 100, data, 255)
    data_inter = np.where(data_inter < threshold, 0, data_inter)
    return np.where((data_inter >= threshold) & (data_inter <= 100), 1, data_inter)


def plot_and_save(file_, cmap_type="tab20c"):
    ## Plotting
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fname_list = f.split(".")[0].split("_")
    contin_char, month_ = fname_list[1], fname_list[5]

    cmap = cm.get_cmap(name=cmap_type)

    fig, ax = plt.subplots(figsize=(10, 5))

    # full path to the geotiff file
    fpath = os.path.join(processing_path, file_)  # Thailand east

    # read extent of image without loading
    # good for values in degrees lat/long
    # geotiff may use other coordinates and projection
    my_image = georaster.SingleBandRaster(fpath, load_data=False)

    # grab limits of image's extent
    minx, maxx, miny, maxy = my_image.extent

    # set Basemap with slightly larger extents
    # set resolution at intermediate level "i"
    m = Basemap(projection='cyl', llcrnrlon=-180, llcrnrlat=0, urcrnrlon=180, urcrnrlat=90, resolution='i')
    m.drawcoastlines(color="gray", linewidth=0.05)
    m.fillcontinents(color='beige')

    # load the geotiff image, assign it a variable
    image = georaster.SingleBandRaster(fpath, load_data=(minx, maxx, miny, maxy), latlon=True)

    # plot the image on matplotlib active axes
    # set zorder to put the image on top of coastlines and continent areas
    # set alpha to let the hidden graphics show through

    monthly_graph = ax.imshow(image.r,
                              extent=(minx, maxx, miny, maxy),
                              cmap=cmap,
                              zorder=10,
                              alpha=0.6,
                              label="A")
    ax.set_title("Monthly Contingency Values of {} for month: {}".format(contin_char, month_))
    ep.colorbar(monthly_graph)
    plt.show()
    plt.savefig(os.path.join(processing_path, file_.split(".")[0] + ".png"), dpi=1200)
    plt.close("all")
    plt.clf()
    del plt
    del image


processing_path = r"/media/knn/F/era5/data"
files = [row for row in glob.glob1(processing_path, "era_*.tif")]
done_list = []

for f in files:
    st = datetime.datetime.now()
    if f not in done_list:
        print(f)
        plot_and_save(f, 'jet')
    print("Duration", datetime.datetime.now() - st)

os.system("taskset -p 0xff %d" % os.getpid())
os.sched_setaffinity(0, {i for i in range(32)})

start = datetime.datetime.now()
h35_mars_path = "/media/knn/DATA/modis_hdf_data/mars_data"
sample_geotiff = r"/media/knn/DATA/modis_hdf_data/merged_daily/MOD10A1_20190218_europe_resampled.tif"

files = [os.path.join(processing_path, row) for row in glob.glob1(processing_path, "*.nc")]

era5_land_data_name = 'adaptor.mars.internal-1613380008.163121-17074-7-0472c0b4-31c9-4617-9658-bac10e1a5621.nc'
era5_land_data = xr.open_dataset(os.path.join(processing_path, era5_land_data_name), chunks={'time': 50})
dates = list(np.datetime_as_string(era5_land_data.coords['time'], unit='D'))
era_validation_data = era5_land_data.sel(time=slice('2018-10-01', '2019-12-31'))
dates = list(np.datetime_as_string(era_validation_data.coords['time'], unit='D'))
snow_threshold_for_mars = 50  # cm
snow_threshold_for_era5 = 0.05  # 5 cm 0.05 m
era5_land_data = era5_land_data.assign_coords(longitude=(era5_land_data.longitude + 180) % 360 - 180).sortby(
    'longitude')
# Define area of interest for the era5 dataset
aoi = era5_land_data.sel(longitude=slice(-180, 180), latitude=slice(90, 0))
# aoi = aoi['sde'].interp(latitude=list(np.arange(90, 0 + 0.01, -0.01)),
#                         longitude=list(np.arange(-180, 180 - 0.01, 0.01)),
#                         method='linear')
error_list = []


def masking_era5(in_data, threshold):
    data = in_data[0]
    return da.where(data >= threshold, 1, 0).astype('b')


# return cloud percentage over land areas (excluding sea or water bodies)
def get_percentage(in_data, cloud_class_value=251, sea_class_value=252):
    from six.moves import reduce
    total_pixels_with_values = reduce(lambda x, y: x * y, in_data.shape)
    sum_clouds = int(np.sum(in_data.where(in_data.values == cloud_class_value).notnull()[0]))
    sum_seas = int(np.sum(in_data.where(in_data.values == sea_class_value).notnull()[0]))
    return sum_clouds / (total_pixels_with_values - sum_seas)


months = [11, 12, 1, 2, 3, 4, 5]
df = pd.DataFrame(columns=["Date",
                           "SumA", "SumB", "SumC", "SumD",
                           "POD", "FAR", "Accuracy",
                           "Start", "Cloud Percentage", "Duration"],
                  index=dates)
df_data = pd.DataFrame(columns=["Month",
                                "A", "B", "C", "D"],
                       index=months)
start = datetime.datetime.now()
prev = 0

for en, day_ in enumerate(dates):
    process_start = datetime.datetime.now()
    print(day_, "is being processed")
    try:
        mars_data = xr.open_rasterio(os.path.join(h35_mars_path, "h35_" + day_.replace("-", "") + "_day_TSMS.tif"))
    except OSError as be:
        error_list.append(day_)
        continue
    month = int(day_.split("-")[1])

    # mars_data_selected = mars_data.sel(x=slice(25, 46), y=slice(45, 34))
    # mars_data_selected = mars_data.sel(x=slice(-180, 180), y=slice(90, 0))
    mars_data_selected = mars_data
    daily_data_era5 = aoi.sel(time=day_)

    # daily_data_era5_linear_interpolated = daily_data_era5['sde'].interp(latitude=list(np.arange(45, 34, -0.01)),
    #                                                                     longitude=list(np.arange(25, 46 + 0.01, 0.01)),
    #                                                                     method='linear')
    daily_data_era5_linear_interpolated = daily_data_era5['sde'].interp(latitude=list(np.arange(90, 0 + 0.01, -0.01)),
                                                                        longitude=list(
                                                                            np.arange(-180, 180 - 0.01, 0.01)),
                                                                        method='linear')
    st = datetime.datetime.now()
    try:
        pr = get_percentage(mars_data_selected)
    except OSError:
        error_list.append([day_, 'percentage'])
        continue
    print("Percentage", datetime.datetime.now() - st)
    st = datetime.datetime.now()
    data_inter_era5 = masking_era5(daily_data_era5_linear_interpolated, snow_threshold_for_era5)
    print("masking_era5", datetime.datetime.now() - st)
    st = datetime.datetime.now()
    data_inter_mars = masking(mars_data_selected, snow_threshold_for_mars)
    print("masking_mars", datetime.datetime.now() - st)
    work_mask = (da.where(mars_data_selected <= 100, 1, 0))
    del mars_data_selected

    st = datetime.datetime.now()
    # Hits (A)
    A = (da.where((data_inter_era5 == 1) & (data_inter_mars == 1), 1, 0) * work_mask).astype('b').compute()
    # False Alarms (B)
    B = (da.where((data_inter_era5 == 0) & (data_inter_mars == 1), 1, 0) * work_mask).astype('b').compute()
    # Misses (C)
    C = (da.where((data_inter_era5 == 1) & (data_inter_mars == 0), 1, 0) * work_mask).astype('b').compute()
    # Correct Negatives (D)
    D = (da.where((data_inter_era5 == 0) & (data_inter_mars == 0), 1, 0) * work_mask).astype('b').compute()
    print("ABCD", datetime.datetime.now() - st)
    st = datetime.datetime.now()

    # Summations
    a = np.sum(A)
    b = np.sum(B)
    c = np.sum(C)
    d = np.sum(D)

    # Validation metrics calculations
    pod = a / (a + c)
    far = b / (a + b)
    acc = (a + d) / (a + b + c + d)
    print("pod,a,b,c,d", datetime.datetime.now() - st)
    # Monthly Summation
    st = datetime.datetime.now()
    if prev != month or en == 0:
        # A, B, C, D = [np.zeros((1100, 2101)) for _ in range(4)]
        A, B, C, D = [da.zeros((8999, 35999), chunks=(1000, 1000), dtype='b').compute() for _ in range(4)]
        df_data.loc[month] = [month, A, B, C, D]
    df_data.loc[month] = [month,
                          df_data.loc[month]['A'] + A,
                          df_data.loc[month]['B'] + B,
                          df_data.loc[month]['C'] + C,
                          df_data.loc[month]['D'] + D]

    print("Monthly", datetime.datetime.now() - st)
    del A, B, C, D

    prev = month
    end = datetime.datetime.now()
    process_end = datetime.datetime.now()
    df.loc[day_] = [pd.Timestamp(day_.replace("-", "")), a, b, c, d, pod, far, acc, start, pr,
                    (process_end - process_start).total_seconds()]
    print("Duration", process_end - process_start)
#
# df_data['POD'] = df_data['A'] / (df_data['A'] + df_data['C'])
# df_data['FAR'] = df_data['B'] / (df_data['A'] + df_data['B'])
# df_data['ACC'] = (df_data['B'] + df_data['D']) / (df_data['A'] + df_data['B'] + df_data['C'] + df_data['D'])

era5_land_data.close()

for row in df_data['Month']:
    for key in df_data.keys():
        if key not in ['Month', 'A', 'B', 'C', 'D'] and row is not np.nan:
            da = xr.DataArray(data=df_data[key][row][0], dims=["y", "x"],
                              coords=[daily_data_era5_linear_interpolated.latitude.values,
                                      daily_data_era5_linear_interpolated.longitude.values])
            da.rio.to_raster(
                os.path.join(processing_path, "era_" + str(key) + "_monthly_cont_mat_" + str(row) + ".tif"))
            del da
            print(datetime.datetime.now())

files = [row for row in glob.glob1(processing_path, "era_*.tif")]
for f in files:
    plot_and_save(f, 'jet')

abcd_geotiff_files = [os.path.join(processing_path, row) for row in glob.glob1(processing_path, "era_*.tif")]
months = list(set([int(f.split("_")[5].split(".")[0]) for f in abcd_geotiff_files]))
for month in months:
    print(month, "is being done..")
    A = xr.open_rasterio(
        [f for f in abcd_geotiff_files if f.find("_" + str(month) + ".tif") > 0 and (f.find("_A_") > 0)][0])
    B = xr.open_rasterio(
        [f for f in abcd_geotiff_files if f.find("_" + str(month) + ".tif") > 0 and (f.find("_B_") > 0)][0])
    C = xr.open_rasterio(
        [f for f in abcd_geotiff_files if f.find("_" + str(month) + ".tif") > 0 and (f.find("_C_") > 0)][0])
    D = xr.open_rasterio(
        [f for f in abcd_geotiff_files if f.find("_" + str(month) + ".tif") > 0 and (f.find("_D_") > 0)][0])
    POD = A / (C + A)
    FAR = B / (A + B)
    ACC = A + B / (A + B + C + D)

    POD.rio.to_raster(os.path.join(processing_path, "era_" + str("POD") + "_monthly_cont_mat_" + str(month) + ".tif"))
    del POD
    FAR.rio.to_raster(os.path.join(processing_path, "era_" + str("FAR") + "_monthly_cont_mat_" + str(month) + ".tif"))
    del FAR
    ACC.rio.to_raster(os.path.join(processing_path, "era_" + str("ACC") + "_monthly_cont_mat_" + str(month) + ".tif"))
    del ACC
    A.close()
    B.close()
    C.close()
    D.close()
    print(month, "is done")
    break
