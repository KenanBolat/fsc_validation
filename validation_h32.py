#!/usr/bin/python3
import os
import datetime
import glob
import geos
import rasterio
from rasterio.merge import merge
from collections import Counter
import pandas as pd
import fiona
import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
import rasterio.mask
import psutil
# region plotting modules
import georaster
import matplotlib

matplotlib.use('Agg')
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import matplotlib.cm as cm

# endregion

start = datetime.datetime.now()


class hsaf_validation(object):
    def __init__(self, product_type: str):
        self.start = datetime.datetime.now()
        self.process_path = None
        self.ground_data_location = None
        self.product_data_location = None
        self.end = None
        self.fractional_snow_threshold = 50
        self.snow_measurement_threshold = 2
        self.snow_threshold = self._get_threshold()
        self.product_type = product_type.lower()

    # TODO convert this to decorator or use built in decorator
    def process_time(self):
        if self.end is None:
            self.end = datetime.datetime.now()
        return self.end - self.start

    def _get_threshold(self):
        if self.product_type in ['h35', 'h12']:
            return self.fractional_snow_threshold
        elif self.product_type in ['h10', 'h34']:
            return self.snow_measurement_threshold


# define paths
processing_path = "/media/knn/F/validation_2020/"
processing_path_mgm = os.path.join(processing_path, "tif")
processing_path_h32 = os.path.join(processing_path, "h32/H32")
snow_threshold = 5

with fiona.open("/home/knn/Desktop/over_turkey.shp", "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]


def masking_h32(in_data):
    data_inter = []
    h32_data = in_data[0]
    data_inter = np.where(h32_data == 1, h32_data, 0)
    return data_inter


def masking_mgm(in_data, threshold):
    data_inter = []
    # mars_data = in_data.read(1)
    mgm_data = in_data[0]
    return np.where((mgm_data >= threshold), 1, 0)


mgm_set = list(set([file_.split(".")[0] for file_ in glob.glob1(processing_path_mgm, "*.tif")]))
h32_set = list(set([file_.split("_")[1].split(".")[0] for file_ in glob.glob1(processing_path_h32, "*.tif")]))
mgm_set.sort()
h32_set.sort()

print(len(mgm_set), len(h32_set))
months = [10, 11, 12, 1, 2, 3, 4, 5, 6]
df = pd.DataFrame(columns=["Date",
                           "SumA", "SumB", "SumC", "SumD",
                           "POD", "FAR", "Accuracy",
                           "Start", "Duration"],
                  index=mgm_set)
df_data = pd.DataFrame(columns=["Month",
                                "A", "B", "C", "D"],
                       index=months)
start = datetime.datetime.now()
prev = 0;

for en, date_ in enumerate(mgm_set):
    inner = datetime.datetime.now()
    in_date_format = datetime.datetime.strptime(date_, "%Y-%m-%d")
    month = in_date_format.month
    if prev != month or en == 0:
        A, B, C, D = [np.zeros((1210, 2049)) for row in range(4)]
        df_data.loc[month] = [month, A, B, C, D]

    print(date_, in_date_format)
    # modis_data = rasterio.open(os.path.join(processing_path_modis, "MOD10A1_" + date_ + "_europe_resampled.tif"))
    with rasterio.open(os.path.join(processing_path_mgm, date_ + ".tif")) as src:
        mgm_data, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    try:
        # mars_data = rasterio.open(os.path.join(processing_path_mars, "h35_" + date_ + "_day_TSMS.tif"))

        with rasterio.open(os.path.join(processing_path_h32, "H32_" + date_ + ".tif")) as src:
            h32_data, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            # out_meta = src.meta
    except:
        continue

    data_inter_mgm = masking_mgm(mgm_data, snow_threshold)
    data_inter_h32 = masking_h32(h32_data)

    h = h32_data[0]
    m = mgm_data[0]
    work_mask = (np.where(m > 0, 1, 0) * np.where((h >= 0) & (h <= 3), 1, 0))
    # Hits(A)
    A = np.where((data_inter_mgm == 1) & (data_inter_h32 == 1), 1, 0) * work_mask
    # False Alarms (B)
    B = np.where((data_inter_mgm == 0) & (data_inter_h32 == 1), 1, 0) * work_mask
    # Misses (C)
    C = np.where((data_inter_mgm == 1) & (data_inter_h32 == 0), 1, 0) * work_mask
    # Correct Negatives (D)
    D = np.where((data_inter_mgm == 0) & (data_inter_h32 == 0), 1, 0) * work_mask

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
    prev = month
    end = datetime.datetime.now()
    duration = end - inner
    df.loc[date_] = [pd.Timestamp(date_), a, b, c, d, pod, far, acc, start, duration.total_seconds()]
    print(duration.total_seconds())
    del data_inter_h32, data_inter_mgm
    del A, B, C, D
    del h32_data

np.max(df_data.loc[month]['A'])
df_data['POD'] = df_data['A'] / (df_data['A'] + df_data['C'])
df_data['FAR'] = df_data['B'] / (df_data['A'] + df_data['B'])
df_data['ACC'] = (df_data['B'] + df_data['D']) / (df_data['A'] + df_data['B'] + df_data['C'] + df_data['D'])

out_meta.update({"driver": "GTiff",
                 "height": df_data.loc[11]['A'].shape[0],
                 "width": df_data.loc[11]['A'].shape[1],
                 "transform": out_transform})

for row in df_data['Month']:
    for key in df_data.keys():
        if key not in ['Month', 'A', 'B', 'C', 'D']:
            with rasterio.open(
                    os.path.join(processing_path, "modis_" + str(key) + "_monthly_cont_mat_" + str(row) + ".tif"), "w",
                    **out_meta) as dest:
                print(datetime.datetime.now())
                d = df_data.loc[row][str(key)].astype('u1')
                dest.write(d.reshape((1, d.shape[0], d.shape[1])))


def plot_and_save(file_, cmap_type="tab20c"):
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


files = [row for row in glob.glob1(processing_path, "modis_*.tif")]
for f in files:
    plot_and_save(f)
