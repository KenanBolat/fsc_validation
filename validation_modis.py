# import modules
import os
import datetime
import glob
import rasterio
from rasterio.merge import merge
from collections import Counter
import pandas as pd
import fiona 
import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
start = datetime.datetime.now()
import rasterio.mask

import psutil
psutil.cpu_count()
p = psutil.Process()
p.cpu_affinity()
#define paths
processing_path = "/media/knn/DATA/modis_hdf_data"
processing_path_mgm = "/media/knn/F/validation_2020/"

processing_path_modis = os.path.join(processing_path, "merged_daily")
processing_path_mars = os.path.join(processing_path, "mars_data")
processing_path_mgm_files = os.path.join(processing_path_mgm, "tif")
snow_threshold = 50
# For Snow measurements
mgm_snow_threshold = 5

with fiona.open("/home/knn/Desktop/over_tr.shp", "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]


def masking(in_data, threshold):
    data_inter = []
    # mars_data = in_data.read(1)
    mars_data = in_data[0]
    data_inter = np.where(mars_data <= 100, mars_data, 255)
    data_inter = np.where(data_inter < threshold, 0, data_inter)
    return np.where((data_inter >= threshold) & (data_inter <= 100), 1, data_inter)


def masking_mgm(in_data, threshold):
    data_inter = []
    mgm_data = in_data[0]
    return np.where((mgm_data >= threshold), 1, 0)

modis_set = list(set([file_.split("_")[1] for file_ in glob.glob1(processing_path_modis, "*.tif")]))
mars_set = list(set([file_.split("_")[1] for file_ in glob.glob1(processing_path_mars, "*.tif")]))
mgm_set = list(set([file_.split(".")[0].replace("-","") for file_ in glob.glob1(processing_path_mgm_files, "*.tif")]))
modis_set.sort()
mars_set.sort()
mgm_set.sort()

print(len(modis_set), len(mars_set))
months = [11, 12, 1, 2, 3, 4, 5]
df = pd.DataFrame(columns=["Date",
                           "SumA", "SumB", "SumC", "SumD",
                           "POD", "FAR", "Accuracy",
                           "Start", "Duration"],
                  index=modis_set)
df_data = pd.DataFrame(columns=["Month",
                                "A", "B", "C", "D"],
                       index=months)
start = datetime.datetime.now()
prev = 0;


for en, date_ in enumerate(modis_set):
    inner = datetime.datetime.now()
    in_date_format = datetime.datetime.strptime(date_, "%Y%m%d")
    month = in_date_format.month
    if prev != month or en == 0:
        A, B, C, D = [np.zeros((865, 2090)) for row in range(4)]
        df_data.loc[month] = [month, A, B, C, D]

    print(date_, in_date_format)
    # modis_data = rasterio.open(os.path.join(processing_path_modis, "MOD10A1_" + date_ + "_europe_resampled.tif"))
    # with rasterio.open(os.path.join(processing_path_modis, "MOD10A1_" + date_ + "_europe_resampled.tif")) as src:
    #     modis_data, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    #     out_meta = src.meta
    date_mgm = date_[0:4]+"-"+date_[4:6]+"-"+date_[6:8]
    try:
        with rasterio.open(os.path.join(processing_path_mgm_files, date_mgm + ".tif")) as src:
            mgm_data, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta
    except:
        continue

    try:

        # mars_data = rasterio.open(os.path.join(processing_path_mars, "h35_" + date_ + "_day_TSMS.tif"))

        with rasterio.open(os.path.join(processing_path_mars, "h35_" + date_ + "_day_TSMS.tif")) as src:
            mars_data, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            # out_meta = src.meta
    except:
        continue

    data_inter_mars = masking(mars_data, snow_threshold)
    # data_inter_modis = masking(modis_data, snow_threshold)
    data_inter_mgm = masking_mgm(mgm_data, mgm_snow_threshold)

    h = mars_data[0]
    m = mgm_data[0]
    work_mask = (np.where(m > 0, 1, 0) )
    # For MODIS
    # # Hits(A)
    # A = np.where((data_inter_modis == 1) & (data_inter_mars == 1), 1, 0)
    # # False Alarms (B)
    # B = np.where((data_inter_modis == 0) & (data_inter_mars == 1), 1, 0)
    # # Misses (C)
    # C = np.where((data_inter_modis == 1) & (data_inter_mars == 0), 1, 0)
    # # Correct Negatives (D)
    # D = np.where((data_inter_modis == 0) & (data_inter_mars == 0), 1, 0)
    #
    # For MGM
    # Hits(A)
    A = np.where((data_inter_mgm == 1) & (data_inter_mars == 1), 1, 0)*work_mask
    # False Alarms (B)
    B = np.where((data_inter_mgm == 0) & (data_inter_mars == 1), 1, 0)*work_mask
    # Misses (C)
    C = np.where((data_inter_mgm == 1) & (data_inter_mars == 0), 1, 0)*work_mask
    # Correct Negatives (D)
    D = np.where((data_inter_mgm == 0) & (data_inter_mars == 0), 1, 0)*work_mask

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
    del data_inter_mars, data_inter_mgm
    del A, B, C, D
    del mars_data


np.max(df_data.loc[month]['A'])
df_data['POD'] = df_data['A']/(df_data['A']  + df_data['C'] )
df_data['FAR'] = df_data['B']/(df_data['A']  + df_data['B'] )
df_data['ACC'] = (df_data['B'] + df_data['D'] )/(df_data['A']  + df_data['B'] +df_data['C']  + df_data['D'] )

out_meta.update({"driver": "GTiff",
                 "height": df_data.loc[11]['A'].shape[0],
                 "width": df_data.loc[11]['A'].shape[1],
                 "transform": out_transform})

for row in df_data['Month']:
    for key in df_data.keys():
        if key not in  ['Month', 'A','B','C','D']:
            with rasterio.open(os.path.join(processing_path, "mgm_" + str(key) + "_monthly_cont_mat_" + str(row) + ".tif"), "w",
                               **out_meta) as dest:
                print(datetime.datetime.now())
                d = df_data.loc[row][str(key)].astype('u2')
                dest.write(d.reshape((1, d.shape[0], d.shape[1])))


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
    m = Basemap( projection='cyl', \
                llcrnrlon=minx-2, \
                llcrnrlat=miny-2, \
                urcrnrlon=maxx+2, \
                urcrnrlat=maxy+2, \
                resolution='i')

    m.drawcoastlines(color="gray")
    m.fillcontinents(color='beige')

    # load the geotiff image, assign it a variable
    image = georaster.SingleBandRaster( fpath, \
                            load_data=(minx, maxx, miny, maxy), \
                            latlon=True)

    # plot the image on matplotlib active axes
    # set zorder to put the image on top of coastlines and continent areas
    # set alpha to let the hidden graphics show through


    norm = colors.BoundaryNorm([0, 2,6, 8, 10, 12,  15,20,25,30], 10)
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
files = [row for row in glob.glob1(processing_path, "mgm_*.tif")]
for f in files:
    plot_and_save(f)