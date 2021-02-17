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

os.system("taskset -p 0xff %d" % os.getpid())
os.sched_setaffinity(0, {i for i in range(32)})

start = datetime.datetime.now()
processing_path = r"/media/knn/F/era5/data"
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


def masking(in_data, threshold):
    data_inter = []
    # mars_data = in_data.read(1)
    data = in_data[0].values
    data_inter = np.where(data <= 100, data, 255)
    data_inter = np.where(data_inter < threshold, 0, data_inter)
    return np.where((data_inter >= threshold) & (data_inter <= 100), 1, data_inter)


def plot_and_save(file_, cmap_type="tab20c"):
    ## Plotting
    import georaster
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from matplotlib.colors import ListedColormap
    import matplotlib.colors as colors
    import earthpy as et
    import earthpy.spatial as es
    import earthpy.plot as ep

    import matplotlib.cm as cm

    fname_list = f.split(".")[0].split("_")
    contin_char, month_ = fname_list[1], fname_list[5]

    # cmap = ListedColormap(["white", "tan", "springgreen", "darkgreen"])
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
    m = Basemap(projection='cyl', \
                llcrnrlon=minx - 2, \
                llcrnrlat=miny - 2, \
                urcrnrlon=maxx + 2, \
                urcrnrlat=maxy + 2, \
                resolution='i')

    m.drawcoastlines(color="gray")
    m.fillcontinents(color='beige')

    # load the geotiff image, assign it a variable
    image = georaster.SingleBandRaster(fpath, \
                                       load_data=(minx, maxx, miny, maxy), \
                                       latlon=True)

    # plot the image on matplotlib active axes
    # set zorder to put the image on top of coastlines and continent areas
    # set alpha to let the hidden graphics show through

    norm = colors.BoundaryNorm([0, 2, 6, 8, 10, 12, 15, 20, 25, 30], 10)
    monthly_graph = ax.imshow(image.r,
                              extent=(minx, maxx, miny, maxy),
                              cmap=cmap,
                              # norm=norm,
                              zorder=10,
                              alpha=0.6,
                              label="A")
    ax.set_title("Monthly Contingency Values of {} for month: {}".format(contin_char, month_))
    cbar = ep.colorbar(monthly_graph)
    boundary_means = [np.mean([norm.boundaries[ii], norm.boundaries[ii - 1]])
                      for ii in range(1, len(norm.boundaries))]
    category_names = ['Low', 'Medium', 'High', 'Maximum']
    plt.show()
    plt.savefig(os.path.join(processing_path, file_.split(".")[0] + ".png"))


print("a")


def masking_era5(in_data, threshold):
    data = in_data[0].values
    return np.where((data >= threshold), 1, 0)


# return cloud percentage over land areas (excluding sea or water bodies)
def get_percentage(in_data, cloud_class_value=251, sea_class_value=252):
    from six.moves import reduce
    total_pixels_with_values = reduce(lambda x, y: x * y, in_data.shape)
    sum_clouds = np.sum(in_data.where(in_data.values == cloud_class_value).notnull()[0])
    sum_seas = np.sum(in_data.where(in_data.values == sea_class_value).notnull()[0])
    return ((sum_clouds) / (total_pixels_with_values - sum_seas)) * 100


months = [10, 11, 12, 1, 2, 3, 4, 5, 6]
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
    except BaseException as be:
        error_list.append(day_)
        continue
    month = int(day_.split("-")[1])

    if prev != month or en == 0:
        A, B, C, D = [np.zeros((1100, 2101)) for row in range(4)]
        # A, B, C, D = [np.zeros((8999, 35999)) for row in range(4)]
        df_data.loc[month] = [month, A, B, C, D]

    mars_data_selected = mars_data.sel(x=slice(25, 46), y=slice(45, 34))
    # mars_data_selected = mars_data.sel(x=slice(-180, 180), y=slice(90, 0))
    # mars_data_selected = mars_data
    daily_data_era5 = aoi.sel(time=day_)

    daily_data_era5_linear_interpolated = daily_data_era5['sde'].interp(latitude=list(np.arange(45, 34, -0.01)),
                                                                        longitude=list(np.arange(25, 46 + 0.01, 0.01)),
                                                                        method='linear')
    # daily_data_era5_nearest_interpolated = daily_data_era5['sde'].interp(latitude=list(np.arange(45, 34, -0.01)),
    #                                                                      longitude=list(np.arange(25, 46 + 0.01, 0.01)),
    #                                                                      method='nearest')
    # daily_data_era5_linear_interpolated = daily_data_era5['sde'].interp(
    #     latitude=list(np.arange(90, 0 + 0.01, -0.01)),
    #     longitude=list(np.arange(-180, 180 - 0.01, 0.01)),
    #     method='linear')
    # daily_data_era5_nearest_interpolated = daily_data_era5['sde'].interp(latitude=list(np.arange(45, 34, -0.01)),
    #                                                                      longitude=list(np.arange(25, 46 + 0.01, 0.01)),
    #                                                                      method='nearest')

    try:
        pr = get_percentage(mars_data_selected)
    except:
        error_list.append([day_, 'percentage'])
        continue
    data_inter_era5 = masking_era5(daily_data_era5_linear_interpolated, snow_threshold_for_era5)

    data_inter_mars = masking(mars_data_selected, snow_threshold_for_mars)
    work_mask = (np.where(mars_data_selected <= 100, 1, 0))
    del mars_data_selected
    A = np.where((data_inter_era5 == 1) & (data_inter_mars == 1), 1, 0) * work_mask
    # False Alarms (B)
    B = np.where((data_inter_era5 == 0) & (data_inter_mars == 1), 1, 0) * work_mask
    # Misses (C)
    C = np.where((data_inter_era5 == 1) & (data_inter_mars == 0), 1, 0) * work_mask
    # Correct Negatives (D)
    D = np.where((data_inter_era5 == 0) & (data_inter_mars == 0), 1, 0) * work_mask
    a = np.sum(A)
    b = np.sum(B)
    c = np.sum(C)
    d = np.sum(D)

    pod = a / (a + c)
    far = b / (a + b)
    acc = (a + d) / (a + b + c + d)

    #
    df_data.loc[month] = [month,
                          df_data.loc[month]['A'] + A,
                          df_data.loc[month]['B'] + B,
                          df_data.loc[month]['C'] + C,
                          df_data.loc[month]['D'] + D]
    del A, B, C, D
    prev = month
    end = datetime.datetime.now()
    process_end = datetime.datetime.now()
    df.loc[day_] = [pd.Timestamp(day_.replace("-", "")), a, b, c, d, pod, far, acc, start, pr,
                    (process_end - process_start).total_seconds()]
    print("Duration", process_end - process_start)

df_data['POD'] = df_data['A'] / (df_data['A'] + df_data['C'])
df_data['FAR'] = df_data['B'] / (df_data['A'] + df_data['B'])
df_data['ACC'] = (df_data['B'] + df_data['D']) / (df_data['A'] + df_data['B'] + df_data['C'] + df_data['D'])

era5_land_data.close()

import rioxarray

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
