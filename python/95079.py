# # PS Orthotile and Landsat 8 Crossovers
# 
# Have you ever wanted to compare PS images to Landsat 8 images? Both image collections are made available via the Planet API. However, it takes a bit of work to identify crossovers - that is, images of the same area that were collected within a reasonable time difference of each other. Also, you may be interested in filtering out some imagery, e.g. cloudy images.
# 
# This notebook walks you through the process of finding crossovers between PS Orthotiles and Landsat 8 scenes. In this notebook, we specify 'crossovers' as images that have been taken within 1hr of eachother. This time gap is sufficiently small that we expect the atmospheric conditions won't change much (this assumption doesn't always hold, but is the best we can do for now). We also filter out cloudy images and constrain our search to images collected in 2017, January 1 through August 23.
# 

# Notebook dependencies
from __future__ import print_function

import datetime
import json
import os

import ipyleaflet as ipyl
import ipywidgets as ipyw
from IPython.core.display import HTML
from IPython.display import display
import pandas as pd
from planet import api
from planet.api import filters
from shapely import geometry as sgeom


# ## Define AOI
# 
# Define the AOI as a geojson polygon. This can be done at [geojson.io](http://geojson.io). If you use geojson.io, only copy the single aoi feature, not the entire feature collection.
# 

aoi = {u'geometry': {u'type': u'Polygon', u'coordinates': [[[-121.3113248348236, 38.28911976564886], [-121.3113248348236, 38.34622533958], [-121.2344205379486, 38.34622533958], [-121.2344205379486, 38.28911976564886], [-121.3113248348236, 38.28911976564886]]]}, u'type': u'Feature', u'properties': {u'style': {u'opacity': 0.5, u'fillOpacity': 0.2, u'noClip': False, u'weight': 4, u'color': u'blue', u'lineCap': None, u'dashArray': None, u'smoothFactor': 1, u'stroke': True, u'fillColor': None, u'clickable': True, u'lineJoin': None, u'fill': True}}}


json.dumps(aoi)


# ## Build Request
# 
# Build the Planet API Filter request for the Landsat 8 and PS Orthotile imagery taken in 2017 through August 23.
# 

# define the date range for imagery
start_date = datetime.datetime(year=2017,month=1,day=1)
stop_date = datetime.datetime(year=2017,month=8,day=23)


# filters.build_search_request() item types:
# Landsat 8 - 'Landsat8L1G'
# Sentinel - 'Sentinel2L1C'
# PS Orthotile = 'PSOrthoTile'

def build_landsat_request(aoi_geom, start_date, stop_date):
    query = filters.and_filter(
        filters.geom_filter(aoi_geom),
        filters.range_filter('cloud_cover', lt=5),
        # ensure has all assets, unfortunately also filters 'L1TP'
#         filters.string_filter('quality_category', 'standard'), 
        filters.range_filter('sun_elevation', gt=0), # filter out Landsat night scenes
        filters.date_range('acquired', gt=start_date),
        filters.date_range('acquired', lt=stop_date)
    )

    return filters.build_search_request(query, ['Landsat8L1G'])    
    
    
def build_ps_request(aoi_geom, start_date, stop_date):
    query = filters.and_filter(
        filters.geom_filter(aoi_geom),
        filters.range_filter('cloud_cover', lt=0.05),
        filters.date_range('acquired', gt=start_date),
        filters.date_range('acquired', lt=stop_date)
    )

    return filters.build_search_request(query, ['PSOrthoTile'])

print(build_landsat_request(aoi['geometry'], start_date, stop_date))
print(build_ps_request(aoi['geometry'], start_date, stop_date))


# ## Search Planet API
# 
# The client is how we interact with the planet api. It is created with the user-specific api key, which is pulled from $PL_API_KEY environment variable. Create the client then use it to search for PS Orthotile and Landsat 8 scenes. Save a subset of the metadata provided by Planet API as our 'scene'.
# 

def get_api_key():
    return os.environ['PL_API_KEY']

# quick check that key is defined
assert get_api_key(), "PL_API_KEY not defined."


def create_client():
    return api.ClientV1(api_key=get_api_key())

def search_pl_api(request, limit=500):
    client = create_client()
    result = client.quick_search(request)
    
    # note that this returns a generator
    return result.items_iter(limit=limit)


items = list(search_pl_api(build_ps_request(aoi['geometry'], start_date, stop_date)))
print(len(items))
# uncomment below to see entire metadata for a PS orthotile
# print(json.dumps(items[0], indent=4))
del items

items = list(search_pl_api(build_landsat_request(aoi['geometry'], start_date, stop_date)))
print(len(items))
# uncomment below to see entire metadata for a landsat scene
# print(json.dumps(items[0], indent=4))
del items


# In processing the items to scenes, we are only using a small subset of the [product metadata](https://www.planet.com/docs/spec-sheets/sat-imagery/#product-metadata). 
# 

def items_to_scenes(items):
    item_types = []

    def _get_props(item):
        props = item['properties']
        props.update({
            'thumbnail': item['_links']['thumbnail'],
            'item_type': item['properties']['item_type'],
            'id': item['id'],
            'acquired': item['properties']['acquired'],
            'footprint': item['geometry']
        })
        return props
    
    scenes = pd.DataFrame(data=[_get_props(i) for i in items])
    
    # acquired column to index, it is unique and will be used a lot for processing
    scenes.index = pd.to_datetime(scenes['acquired'])
    del scenes['acquired']
    scenes.sort_index(inplace=True)
    
    return scenes

scenes = items_to_scenes(search_pl_api(build_landsat_request(aoi['geometry'],
                                                             start_date, stop_date)))
# display(scenes[:1])
print(scenes.thumbnail.tolist()[0])
del scenes


# ## Investigate Landsat Scenes
# 
# There are quite a few Landsat 8 scenes that are returned by our query. What do the footprints look like relative to our AOI and what is the collection time of the scenes?

landsat_scenes = items_to_scenes(search_pl_api(build_landsat_request(aoi['geometry'],
                                                                     start_date, stop_date)))

# How many Landsat 8 scenes match the query?
print(len(landsat_scenes))


# ### Show Landsat 8 Footprints on Map
# 

def landsat_scenes_to_features_layer(scenes):
    features_style = {
            'color': 'grey',
            'weight': 1,
            'fillColor': 'grey',
            'fillOpacity': 0.15}

    features = [{"geometry": r.footprint,
                 "type": "Feature",
                 "properties": {"style": features_style,
                                "wrs_path": r.wrs_path,
                                "wrs_row": r.wrs_row}}
                for r in scenes.itertuples()]
    return features

def create_landsat_hover_handler(scenes, label):
    def hover_handler(event=None, id=None, properties=None):
        wrs_path = properties['wrs_path']
        wrs_row = properties['wrs_row']
        path_row_query = 'wrs_path=={} and wrs_row=={}'.format(wrs_path, wrs_row)
        count = len(scenes.query(path_row_query))
        label.value = 'path: {}, row: {}, count: {}'.format(wrs_path, wrs_row, count)
    return hover_handler


def create_landsat_feature_layer(scenes, label):
    
    features = landsat_scenes_to_features_layer(scenes)
    
    # Footprint feature layer
    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    feature_layer = ipyl.GeoJSON(data=feature_collection)

    feature_layer.on_hover(create_landsat_hover_handler(scenes, label))
    return feature_layer


# Initialize map using parameters from above map
# and deleting map instance if it exists
try:
    del fp_map
except NameError:
    pass


zoom = 6
center = [38.28993659801203, -120.14648437499999] # lat/lon


# Create map, adding box drawing controls
# Reuse parameters if map already exists
try:
    center = fp_map.center
    zoom = fp_map.zoom
    print(zoom)
    print(center)
except NameError:
    pass

# Change tile layer to one that makes it easier to see crop features
# Layer selected using https://leaflet-extras.github.io/leaflet-providers/preview/
map_tiles = ipyl.TileLayer(url='http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png')
fp_map = ipyl.Map(
        center=center, 
        zoom=zoom,
        default_tiles = map_tiles
    )

label = ipyw.Label(layout=ipyw.Layout(width='100%'))
fp_map.add_layer(create_landsat_feature_layer(landsat_scenes, label)) # landsat layer
fp_map.add_layer(ipyl.GeoJSON(data=aoi)) # aoi layer
    
# Display map and label
ipyw.VBox([fp_map, label])


# This AOI is located in a region covered by 3 different path/row tiles. This means there is 3x the coverage than in regions only covered by one path/row tile. This is particularly lucky!
# 
# What about the within each path/row tile. How long and how consistent is the Landsat 8 collect period for each path/row?

def time_diff_stats(group):
    time_diff = group.index.to_series().diff() # time difference between rows in group
    stats = {'median': time_diff.median(),
             'mean': time_diff.mean(),
             'std': time_diff.std(),
             'count': time_diff.count(),
             'min': time_diff.min(),
             'max': time_diff.max()}
    return pd.Series(stats)

landsat_scenes.groupby(['wrs_path', 'wrs_row']).apply(time_diff_stats)


# It looks like the collection period is 16 days, which lines up with the [Landsat 8 mission description](https://landsat.usgs.gov/landsat-8).
# 
# path/row 43/33 is missing one image which causes an unusually long collect period.
# 
# What this means is that we don't need to look at every Landsat 8 scene collect time to find crossovers with Planet scenes. We could look at the first scene for each path/row, then look at every 16 day increment. However, we will need to account for dropped Landsat 8 scenes in some way.
# 
# What is the time difference between the tiles?

def find_closest(date_time, data_frame):
    # inspired by:
    # https://stackoverflow.com/questions/36933725/pandas-time-series-join-by-closest-time
    time_deltas = (data_frame.index - date_time).to_series().reset_index(drop=True).abs()
    idx_min = time_deltas.idxmin()

    min_delta = time_deltas[idx_min]
    return (idx_min, min_delta)

def closest_time(group):
    '''group: data frame with acquisition time as index'''
    inquiry_date = datetime.datetime(year=2017,month=3,day=7)
    idx, _ = find_closest(inquiry_date, group)
    return group.index.to_series().iloc[idx]


# for accurate results, we look at the closest time for each path/row tile to a given time
# using just the first entry could result in a longer time gap between collects due to
# the timing of the first entries
landsat_scenes.groupby(['wrs_path', 'wrs_row']).apply(closest_time)


# So the tiles that are in the same path are very close (24sec) together from the same day. Therefore, we would want to only use one tile and pick the best image.
# 
# Tiles that are in different paths are 7 days apart. Therefore, we want to keep tiles from different paths, as they represent unique crossovers.
# 

# ## Investigate PS Orthotiles
# 
# There are also quite a few PS Orthotiles that match our query. Some of those scenes may not have much overlap with our AOI. We will want to filter those out. Also, we are interested in knowing how many unique days of coverage we have, so we will group PS Orthotiles by collect day, since we may have days with more than one collect (due multiple PS satellites collecting imagery).
# 

all_ps_scenes = items_to_scenes(search_pl_api(build_ps_request(aoi['geometry'], start_date, stop_date)))

# How many PS scenes match query?
print(len(all_ps_scenes))
all_ps_scenes[:1]


# What about overlap? We really only want images that overlap over 20% of the AOI.
# 
# Note: we do this calculation in WGS84, the geographic coordinate system supported by geojson. The calculation of coverage expects that the geometries entered are 2D, which WGS84 is not. This will cause a small inaccuracy in the coverage area calculation, but not enough to bother us here.
# 

def aoi_overlap_percent(footprint, aoi):
    aoi_shape = sgeom.shape(aoi['geometry'])
    footprint_shape = sgeom.shape(footprint)
    overlap = aoi_shape.intersection(footprint_shape)
    return overlap.area / aoi_shape.area

overlap_percent = all_ps_scenes.footprint.apply(aoi_overlap_percent, args=(aoi,))
all_ps_scenes = all_ps_scenes.assign(overlap_percent = overlap_percent)
all_ps_scenes.head()


print(len(all_ps_scenes))
ps_scenes = all_ps_scenes[all_ps_scenes.overlap_percent > 0.20]
print(len(ps_scenes))


# Ideally, PS scenes have daily coverage over all regions. How many days have PS coverage and how many PS scenes were taken on the same day?

# Use PS acquisition year, month, and day as index and group by those indices
# https://stackoverflow.com/questions/14646336/pandas-grouping-intra-day-timeseries-by-date
daily_ps_scenes = ps_scenes.index.to_series().groupby([ps_scenes.index.year,
                                                       ps_scenes.index.month,
                                                       ps_scenes.index.day])


daily_count = daily_ps_scenes.agg('count')

# How many days is the count greater than 1?
daily_multiple_count = daily_count[daily_count > 1]

print('out of {} days of coverage, {} days have multiple collects.'.format(     len(daily_count), len(daily_multiple_count)))

daily_multiple_count


def scenes_and_count(group):
    entry = {'count': len(group),
             'acquisition_time': group.index.tolist()}
    
    return pd.DataFrame(entry)

daily_count_and_scenes = daily_ps_scenes.apply(scenes_and_count)

multiplecoverage = daily_count_and_scenes.query('count > 1')

# multiplecoverage  # look at all occurrence
multiplecoverage.query('ilevel_1 == 7')  # look at just occurrence in July


# Looks like the multiple collects on the same day are just a few minutes apart. They are likely crossovers between different PS satellites. Cool! Since we only want to us one PS image for a crossover, we will chose the best collect for days with multiple collects.
# 

# ## Find Crossovers
# 
# Now that we have the PS Orthotiles filtered to what we want and have investigated the Landsat 8 scenes, let's look for crossovers between the two.
# 
# First we find concurrent crossovers, PS and Landsat collects that occur within 1hour of each other.
# 

def find_crossovers(acquired_time, landsat_scenes):
    '''landsat_scenes: pandas dataframe with acquisition time as index'''
    closest_idx, closest_delta = find_closest(acquired_time, landsat_scenes)
    closest_landsat = landsat_scenes.iloc[closest_idx]

    crossover = {'landsat_acquisition': closest_landsat.name,
                 'delta': closest_delta}
    return pd.Series(crossover)


# fetch PS scenes
ps_scenes = items_to_scenes(search_pl_api(build_ps_request(aoi['geometry'],
                                                           start_date, stop_date)))


# for each PS scene, find the closest Landsat scene
crossovers = ps_scenes.index.to_series().apply(find_crossovers, args=(landsat_scenes,))

# filter to crossovers within 1hr
concurrent_crossovers = crossovers[crossovers['delta'] < pd.Timedelta('1 hours')]
print(len(concurrent_crossovers))
concurrent_crossovers


# Now that we have the crossovers, what we are really interested in is the IDs of the landsat and PS scenes, as well as how much they overlap the AOI.
# 

def get_crossover_info(crossovers, aoi):
    def get_scene_info(acquisition_time, scenes):
        scene = scenes.loc[acquisition_time]
        scene_info = {'id': scene.id,
                      'thumbnail': scene.thumbnail,
                      # we are going to use the footprints as shapes so convert to shapes now
                      'footprint': sgeom.shape(scene.footprint)}
        return pd.Series(scene_info)

    landsat_info = crossovers.landsat_acquisition.apply(get_scene_info, args=(landsat_scenes,))
    ps_info = crossovers.index.to_series().apply(get_scene_info, args=(ps_scenes,))

    footprint_info = pd.DataFrame({'landsat': landsat_info.footprint,
                                   'ps': ps_info.footprint})
    overlaps = footprint_info.apply(lambda x: x.landsat.intersection(x.ps),
                                    axis=1)
    
    aoi_shape = sgeom.shape(aoi['geometry'])
    overlap_percent = overlaps.apply(lambda x: x.intersection(aoi_shape).area / aoi_shape.area)
    crossover_info = pd.DataFrame({'overlap': overlaps,
                                   'overlap_percent': overlap_percent,
                                   'ps_id': ps_info.id,
                                   'ps_thumbnail': ps_info.thumbnail,
                                   'landsat_id': landsat_info.id,
                                   'landsat_thumbnail': landsat_info.thumbnail})
    return crossover_info

crossover_info = get_crossover_info(concurrent_crossovers, aoi)
print(len(crossover_info))


# Next, we filter to overlaps that cover a significant portion of the AOI.
# 

significant_crossovers_info = crossover_info[crossover_info.overlap_percent > 0.9]
print(len(significant_crossovers_info))
significant_crossovers_info


# Browsing through the crossovers, we see that in some instances, multiple crossovers take place on the same day. Really, we are interested in 'unique crossovers', that is, crossovers that take place on unique days. Therefore, we will look at the concurrent crossovers by day.
# 

def group_by_day(data_frame):
    return data_frame.groupby([data_frame.index.year,
                               data_frame.index.month,
                               data_frame.index.day])

unique_crossover_days = group_by_day(significant_crossovers_info.index.to_series()).count()
print(len(unique_crossover_days))
print(unique_crossover_days)


# There are 6 unique crossovers between Landsat 8 and PS that cover over 90% of our AOI between January and August in 2017. Not bad! That is definitely enough to perform comparison.
# 

# ### Display Crossovers
# 
# Let's take a quick look at the crossovers we found to make sure that they don't look cloudy, hazy, or have any other quality issues that would affect the comparison.
# 

# https://stackoverflow.com/questions/36006136/how-to-display-images-in-a-row-with-ipython-display
def make_html(image):
     return '<img src="{}" style="display:inline;margin:1px"/>'             .format(image)


def display_thumbnails(row):
    print(row.name)
    display(HTML(''.join(make_html(t)
                         for t in (row.ps_thumbnail, row.landsat_thumbnail))))

_ = significant_crossovers_info.apply(display_thumbnails, axis=1)


# They all look pretty good although the last crossover (2017-08-10) could be a little hazy.
# 




# # Segmentation: KNN
# 
# Our aim in this notebook is to automatically identify crops in a Planet image and then create georeferenced geojson features that represent those crops.
# 
# In this notebook, we train a KNN classifier to predict crop/non-crop pixels on one Planet image based on 2-year-old ground truth data, and we use that classifier to predict crop/non-crop pixels in another Planet image. We then segment the classified image and create georeferenced geojson features that outline the predicted crops.
# 
# K-nearest neighbors (KNN) is a supervised learning technique for classification. It is a straightforward machine-learning algorithm. See [wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) for more details. In this notebook, we use KNN Classification to classify pixels as crop / non-crop based on their red, green, blue, and NIR band top-of-atmosphere (TOA) reflectance values.
# 
# TOA reflectance is not affected by the satellite that takes the image, the time of day, or the time of year. It is affected by atmosphere, which isn't ideal. However, correcting for the atmosphere is complicated, requires detailed input data, and is error-prone. Therefore, TOA reflectance is the dataset we are going to use. The Planet product we use has the bands scaled to TOA radiance. We use the radiance coefficient, stored in the product metadata, to scale bands to TOA reflectance.
# 
# The data used in this notebook is selected and described in the [Identify Datasets](datasets-identify.ipynb) notebook and prepared in the [Prepare Datasets](datasets-prepare.ipynb) notebook. Survey data from 2015 is used as the 'ground truth' for training the KNN Classifier and assessing accuracy. The Planet imagery used in this notebook was collected in 2017, so we expect there to be some inaccuracies in the ground truth data due to the 2-year gap. However, we don't expect a large difference in the crop/non-crop status of regions, so expect reasonable accuracy.
# 

# Notebook dependencies
from __future__ import print_function

from collections import namedtuple
import copy
from functools import partial
import json
import os
from xml.dom import minidom

import cv2
import ipyleaflet as ipyl
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from shapely.geometry import mapping, shape, Polygon
from shapely.ops import transform
from sklearn import neighbors
from sklearn.metrics import classification_report

get_ipython().magic('matplotlib inline')


# ### Input Data
# 
# We define the datasets to use for training the KNN classifier and testing the classifier. The training dataset will actually come from the 'test' folder, the test dataset will come from the 'train' folder. The reason for this swap is because the dataset in the 'train' folder is mostly composed of 'crop' regions while the datasets in the 'test' folder are composed of an approximately equal representation of 'crop' and 'non-crop' regions. Swapping these datasets results in an improvement in the classifier f-score on the test dataset of almost 10% (from 0.68 to 0.77 in the trial run).
# 

# Train data
train_data_folder = os.path.join('data', 'test')

# Created in datasets-identify notebook
train_pl_metadata_filename = os.path.join(train_data_folder, '20160831_180257_0e26_3B_AnalyticMS_metadata.xml')
assert os.path.isfile(train_pl_metadata_filename)

# Created in datasets-prepare notebook
train_pl_filename = os.path.join(train_data_folder, '20160831_180257_0e26_3B_AnalyticMS_cropped.tif')
assert os.path.isfile(train_pl_filename)

train_ground_truth_filename = os.path.join(train_data_folder, 'ground-truth_cropped.geojson')
assert os.path.isfile(train_ground_truth_filename)


# Test data
test_data_folder = os.path.join('data', 'train')

# Created in datasets-identify notebook
test_pl_metadata_filename = os.path.join(test_data_folder, '20160831_180231_0e0e_3B_AnalyticMS_metadata.xml')
assert os.path.isfile(test_pl_metadata_filename)

# Created in datasets-prepare notebook
test_pl_filename = os.path.join(test_data_folder, '20160831_180231_0e0e_3B_AnalyticMS_cropped.tif')
assert os.path.isfile(test_pl_filename)

test_ground_truth_filename = os.path.join(test_data_folder, 'ground-truth_cropped.geojson')
assert os.path.isfile(test_ground_truth_filename)


# ## KNN Classification
# 
# Steps:
# 1. Training KNN Classifier
#  - load and convert ground truth data to contours
#  - load image and convert to reflectance
#  - use contours to get crop / non-crop pixels
#  - use crop / non-crop pixels as knn train  
# 2. Testing KNN Classifier
#  - predict crop / non-crop pixels on same image
#  - predict crop / non-crop pixels on train image
# 
# ### Train Classifier
# 
# To train the KNN classifier, we need to identify the class of each pixel as either 'crop' or 'non-crop'. The class is determined from whether the pixel is associated with a crop feature in the ground truth data. To make this determination, we must first convert the ground truth data into OpenCV contours, then we use these contours to separate 'crop' from 'non-crop' pixels. The pixel values and their classification are then used to train the KNN classifier.
# 

# #### Features to Contours
# 
# To convert the ground truth features to contours, we must first project the features to the coordinate reference system of the image, transform the features into image space, and finally convert the features in image space to contours.
# 

# Utility functions: features to contours

def project_feature(feature, proj_fcn):
    """Creates a projected copy of the feature.
    
    :param feature: geojson description of feature to be projected
    :param proj_fcn: partial function defining projection transform"""
    g1 = shape(feature['geometry'])
    g2 = transform(proj_fcn, g1)
    proj_feat = copy.deepcopy(feature)
    proj_feat['geometry'] = mapping(g2)
    return proj_feat


def project_features_to_srs(features, img_srs, src_srs='epsg:4326'):
    """Project features to img_srs.
    
    If src_srs is not specified, WGS84 (only geojson-supported crs) is assumed.
    
    :param features: list of geojson features to be projected
    :param str img_srs: destination spatial reference system
    :param str src_srs: source spatial reference system
    """
    # define projection
    # from shapely [docs](http://toblerity.org/shapely/manual.html#shapely.ops.transform)
    proj_fcn = partial(
        pyproj.transform,
        pyproj.Proj(init=src_srs),
        pyproj.Proj(init=img_srs))
    
    return [project_feature(f, proj_fcn) for f in features]


def polygon_to_contour(feature_geometry, image_transform):
    """Convert the exterior ring of a geojson Polygon feature to an
    OpenCV contour.
    
    image_transform is typically obtained from `img.transform` where 
    img is obtained from `rasterio.open()
    
    :param feature_geometry: the 'geometry' entry in a geojson feature
    :param rasterio.Affine image_transform: image transformation"""
    points_xy = shape(feature_geometry).exterior.coords
    points_x, points_y = zip(*points_xy) # inverse of zip
    rows, cols = rasterio.transform.rowcol(image_transform, points_x, points_y)
    return np.array([pnt for pnt in zip(cols, rows)], dtype=np.int32)


def load_geojson(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def load_contours(ground_truth_filename, pl_filename):
    with rasterio.open(pl_filename) as img:
        img_transform = img.transform
        img_srs = img.crs['init']

    ground_truth_data = load_geojson(ground_truth_filename)

    # project to image srs
    projected_features = project_features_to_srs(ground_truth_data, img_srs)

    # convert projected features to contours
    contours = [polygon_to_contour(f['geometry'], img_transform)
                for f in projected_features]
    return contours

print(len(load_contours(train_ground_truth_filename, train_pl_filename)))


# #### Visualize Contours over Image
# 
# To ensure our contours are accurate, we will visualize them overlayed on the image.
# 

# Utility functions: loading an image

NamedBands = namedtuple('NamedBands', 'b, g, r, nir')

def load_bands(filename):
    """Loads a 4-band BGRNir Planet Image file as a list of masked bands.
    
    The masked bands share the same mask, so editing one band mask will
    edit them all."""
    with rasterio.open(filename) as src:
        b, g, r, nir = src.read()
        mask = src.read_masks(1) == 0 # 0 value means the pixel is masked
    
    bands = NamedBands(b=b, g=g, r=r, nir=nir)
    return NamedBands(*[np.ma.array(b, mask=mask)
                        for b in bands])

def get_rgb(named_bands):
    return [named_bands.r, named_bands.g, named_bands.b]

def check_mask(band):
    return '{}/{} masked'.format(band.mask.sum(), band.mask.size)
    
print(check_mask(load_bands(train_pl_filename).r))


# Utility functions: converting an image to reflectance

NamedCoefficients = namedtuple('NamedCoefficients', 'b, g, r, nir')

def read_refl_coefficients(metadata_filename):
    # https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/ndvi/ndvi_planetscope.ipynb
    xmldoc = minidom.parse(metadata_filename)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)
    return NamedCoefficients(b=coeffs[1],
                             g=coeffs[2],
                             r=coeffs[3],
                             nir=coeffs[4])


def to_reflectance(bands, coeffs):
    return NamedBands(*[b.astype(float)*c for b,c in zip(bands, coeffs)])


def load_refl_bands(filename, metadata_filename):
    bands = load_bands(filename)
    coeffs = read_refl_coefficients(metadata_filename)
    return to_reflectance(bands, coeffs)


print(read_refl_coefficients(train_pl_metadata_filename))


# Utility functions: displaying an image

def _linear_scale(ndarray, old_min, old_max, new_min, new_max):
    """Linear scale from old_min to new_min, old_max to new_max.
    
    Values below min/max are allowed in input and output.
    Min/Max values are two data points that are used in the linear scaling.
    """
    #https://en.wikipedia.org/wiki/Normalization_(image_processing)
    return (ndarray - old_min)*(new_max - new_min)/(old_max - old_min) + new_min
# print(linear_scale(np.array([1,2,10,100,256,2560, 2660]), 2, 2560, 0, 256))

def _mask_to_alpha(bands):
#     band = np.atleast_3d(bands)[...,0]
    band = np.atleast_3d(bands[0])
    alpha = np.zeros_like(band)
    alpha[~band.mask] = 1
    return alpha

def _add_alpha_mask(bands):
    return np.dstack([bands, _mask_to_alpha(bands)])

def bands_to_display(bands, alpha=False):
    """Converts a list of 3 bands to a 3-band rgb, normalized array for display."""  
    assert len(bands) in [1,3]
    all_bands = np.dstack(bands)
    old_min = np.percentile(all_bands, 2)
    old_max = np.percentile(all_bands, 98)

    new_min = 0
    new_max = 1
    scaled = [np.clip(_linear_scale(b.astype(np.double),
                                    old_min, old_max, new_min, new_max),
                      new_min, new_max)
              for b in bands]

    filled = [b.filled(fill_value=new_min) for b in scaled]
    if alpha:
        filled.append(_mask_to_alpha(scaled))

    return np.dstack(filled)


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15,15))
for ax in (ax1, ax2):
    ax.set_adjustable('box-forced')

ax1.imshow(bands_to_display(get_rgb(load_bands(train_pl_filename)), alpha=False))
ax1.set_title('Display Bands, No Alpha')

ax2.imshow(bands_to_display(get_rgb(load_bands(train_pl_filename)), alpha=True))
ax2.set_title('Display Bands, Alpha from Mask')
plt.tight_layout()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15,15))
for ax in (ax1, ax2):
    ax.set_adjustable('box-forced')
    
ax1.imshow(bands_to_display(get_rgb(load_refl_bands(train_pl_filename, train_pl_metadata_filename)), alpha=False))
ax1.set_title('Reflectance Bands, No Alpha')

ax2.imshow(bands_to_display(get_rgb(load_refl_bands(train_pl_filename, train_pl_metadata_filename)), alpha=True))
ax2.set_title('Reflectance Bands, Alpha from Mask')
plt.tight_layout()


# Utility functions: displaying contours

def draw_contours(img, contours, color=(0, 1, 0), thickness=2):
    """Draw contours over a copy of the image"""
    n_img = img.copy()
    # http://docs.opencv.org/2.4.2/modules/core/doc/drawing_functions.html#drawcontours
    cv2.drawContours(n_img,contours,-1,color,thickness=thickness)
    return n_img


plt.figure(1, figsize=(10,10))
plt.imshow(draw_contours(bands_to_display(get_rgb(load_bands(train_pl_filename)), alpha=False),
                         load_contours(train_ground_truth_filename, train_pl_filename)))
_ = plt.title('Contours Drawn over Image')


# #### Separate Crop and Non-Crop Pixels
# 
# To train the knn classifier, we need to separate crop from non-crop pixels. We will do this by using the crop feature contours to mask the crop or non-crop pixels.
# 

# Utility functions: masking pixels using contours

def _create_contour_mask(contours, shape):
    """Masks out all pixels that are not within a contour"""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (1), thickness=-1)
    return mask < 1

def combine_masks(masks):
    """Masks any pixel that is masked in any mask.
    
    masks is a list of masks, all the same size"""
    return np.any(np.dstack(masks), 2)

def _add_mask(bands, mask):
    # since band masks are linked, could use any band here
    bands[0].mask = combine_masks([bands[0].mask, mask])

def mask_contours(bands, contours, in_contours=False):
    contour_mask = _create_contour_mask(contours, bands[0].mask.shape)
    if in_contours:
        contour_mask = ~contour_mask
    _add_mask(bands, contour_mask)
    return bands


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(15,15))
for ax in (ax1, ax2):
    ax.set_adjustable('box-forced')

ax1.imshow(bands_to_display(get_rgb(mask_contours(load_bands(train_pl_filename),
                                                  load_contours(train_ground_truth_filename,
                                                                train_pl_filename))),
                            alpha=True))
ax1.set_title('Crop Pixels')

ax2.imshow(bands_to_display(get_rgb(mask_contours(load_bands(train_pl_filename),
                                                  load_contours(train_ground_truth_filename,
                                                                train_pl_filename),
                                                  in_contours=True)),
                            alpha=True))
ax2.set_title('Non-Crop Pixels')
plt.tight_layout()


# #### Fit the classifier
# 
# We use the scikit-learn [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) to perform KNN classification.
# 
# KNeighborsClassifier fit input is two datasets: X, a 2d array that provides the features to be classified on, and y, a 1d array that provides the classes. X and y are ordered along the first dimension ([example](http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py)).
# 
# The number of neighbors (the K in KNN) is a parameter of the estimator that can greatly affect it's performance. Instead of guessing the parameter, we use cross-validation to tune the parameter. Cross-validation is performed in [KNN Cross-Validation](segment-knn-cross-val.ipynb). In this notebook, we save the X and y datasets so they can be imported by that notebook and we use the number of neighbors that was identified in that notebook to provide the greatest accuracy.
# 

# We first take the non-crop and crop datasets above and create a single classified band, where the values of the band indicate the pixel class.
# 

def classified_band_from_masks(band_mask, class_masks):
    class_band = np.zeros(band_mask.shape, dtype=np.uint8)

    for i, mask in enumerate(class_masks):
        class_band[~mask] = i

    # turn into masked array, using band_mask as mask
    return np.ma.array(class_band, mask=band_mask)


def create_contour_classified_band(pl_filename, ground_truth_filename):
    band_mask = load_bands(pl_filename)[0].mask
    contour_mask = _create_contour_mask(load_contours(ground_truth_filename, pl_filename),
                                        band_mask.shape)
    return classified_band_from_masks(band_mask, [contour_mask, ~contour_mask])

plt.figure(1, figsize=(10,10))
plt.imshow(create_contour_classified_band(train_pl_filename, train_ground_truth_filename))
_ = plt.title('Ground Truth Classes')


# Next we convert this band and the underlying image pixel band values to the X and y inputs to the classifier fit function.
# 

def to_X(bands):
    """Convert to a list of pixel values, maintaining order and filtering out masked pixels."""
    return np.stack([b.compressed() for b in bands], axis=1)

def to_y(classified_band):
    return classified_band.compressed()


# We then save X and y for import into [KNN Parameter Tuning](segment-knn-tuning.ipynb) and then specify the number of neighbors that was found in that notebook to provide the best accuracy.
# 

def save_cross_val_data(pl_filename, ground_truth_filename, metadata_filename):
    train_class_band = create_contour_classified_band(pl_filename, ground_truth_filename)
    X = to_X(load_refl_bands(pl_filename, metadata_filename))
    y = to_y(train_class_band)

    cross_val_dir = os.path.join('data', 'knn_cross_val')
    if not os.path.exists(cross_val_dir):
        os.makedirs(cross_val_dir)

    xy_file = os.path.join(cross_val_dir, 'xy_file.npz')
    np.savez(xy_file, X=X, y=y)
    return xy_file

print(save_cross_val_data(train_pl_filename,
                          train_ground_truth_filename,
                          train_pl_metadata_filename))


# best n_neighbors found in KNN Parameter Tuning
n_neighbors = 3


# Finally, we fit the classifier.
# 

def fit_classifier(pl_filename, ground_truth_filename, metadata_filename, n_neighbors):
    n_neighbors = 15
    weights = 'uniform'
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    
    train_class_band = create_contour_classified_band(pl_filename, ground_truth_filename)
    X = to_X(load_refl_bands(pl_filename, metadata_filename))
    y = to_y(train_class_band)
    clf.fit(X, y)
    return clf

clf = fit_classifier(train_pl_filename,
                     train_ground_truth_filename,
                     train_pl_metadata_filename,
                     n_neighbors)


# #### Test Fit on Train Data
# 
# To see how well the classifier works on its own input data, we use the classifier to predict the class of pixels using just the pixel band values. How well the classifier performs is based mostly on (1) the accuracy of the train data and (2) clear separation of the pixel classes based on the input data (pixel band values). The classifier parameters (number of neighbors, distance weighting function) can be tweaked to overcome weaknesses in (2).
# 

def classified_band_from_y(band_mask, y):
    class_band = np.ma.array(np.zeros(band_mask.shape),
                             mask=band_mask.copy())
    class_band[~class_band.mask] = y
    return class_band


def predict(pl_filename, metadata_filename, clf):
    bands = load_refl_bands(pl_filename, metadata_filename)
    X = to_X(bands)

    y = clf.predict(X)
    
    return classified_band_from_y(bands[0].mask, y)

# it takes a while to run the prediction so cache the results
train_predicted_class_band = predict(train_pl_filename, train_pl_metadata_filename, clf)


def imshow_class_band(ax, class_band):
    """Show classified band with legend. Alters ax in place."""
    im = ax.imshow(class_band)

    # https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
    colors = [ im.cmap(im.norm(value)) for value in (0,1)]
    labels = ('crop', 'non-crop')

    # https://matplotlib.org/users/legend_guide.html
    # tag: #creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
    patches = [mpatches.Patch(color=c, label=l) for c,l in zip(colors, labels)]
    
    ax.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.)

def plot_predicted_vs_truth(predicted_class_band, truth_class_band, figsize=(15,15)):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   sharex=True, sharey=True,
                                   figsize=figsize)
    for ax in (ax1, ax2):
        ax.set_adjustable('box-forced')

    imshow_class_band(ax1, predicted_class_band)
    ax1.set_title('Classifier Predicted Classes')

    imshow_class_band(ax2, truth_class_band)
    ax2.set_title('Ground Truth Classes')
    plt.tight_layout()

plot_predicted_vs_truth(train_predicted_class_band,
                        create_contour_classified_band(train_pl_filename,
                                                       train_ground_truth_filename))


# The predicted classes are close to the ground truth classes, but appear a little noisier and there are a few regions that are different.
# 
# There is one instance where they disagree, the crop region in rows 1650-2150, columns 1500-2000.
# 

# Utility functions: Comparing ground truth and actual in a region

def calculate_ndvi(bands):
    return (bands.nir.astype(np.float) - bands.r) / (bands.nir + bands.r)

def plot_region_compare(region_slice, predicted_class_band, class_band, img_bands):
    # x[1:10:5,::-1] is equivalent to 
    # obj = (slice(1,10,5), slice(None,None,-1)); x[obj]
    # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                                                 figsize=(10,10))
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_adjustable('box-forced')

    imshow_class_band(ax1, predicted_class_band[region_slice])
    ax1.set_title('Predicted Classes')
    
    imshow_class_band(ax2, class_band[region_slice])
    ax2.set_title('Ground Truth Classes')

    region_bands = NamedBands(*[b[region_slice] for b in img_bands])

    ax3.imshow(bands_to_display(get_rgb(region_bands)))
    ax3.set_title('Planet Scene RGB')

    ax4.imshow(bands_to_display(3*[calculate_ndvi(region_bands)], alpha=False))
    ax4.set_title('Planet Scene NDVI')
    
    plt.tight_layout()


plot_region_compare((slice(1650,2150,None), slice(600,1100,None)),
                    train_predicted_class_band,
                    create_contour_classified_band(train_pl_filename,
                                                   train_ground_truth_filename),
                    load_refl_bands(train_pl_filename, train_pl_metadata_filename))


# The classifier filters out the regions identified in the ground truth as 'crop' on the right side of this region. The region is brown in the RGB image and doesn't appear to be a crop. However, in the center of the region, the classifier agrees with the ground truth that this region, which appears light brown in RGB and dark in NDVI, is a 'crop'. It is unclear from visual inspection why this region is classified as a crop, and without ground truth data from the year the image was taken, it is hard to verify whether this classification is correct or not. However, it appears to be a misclassification due to errors in the ground truth data (ie the ground truth data indicates this is a 'crop' region while it is not) due to the time gap between when the ground truth data was collected and when the imagery was collected.
# 

# ### Test Classifier
# 
# Use the KNN classifier trained on the training data to predict the crop regions in the test data.
# 

# First, let's check out a 'noisier' region. We will zoom into rows 600-1000, columns 200-600 to take a look.
# 

plt.figure(1, figsize=(8,8))
plt.imshow(bands_to_display(get_rgb(load_refl_bands(test_pl_filename, test_pl_metadata_filename)),
                            alpha=True))
_ = plt.title('Test Image RGB')


test_predicted_class_band = predict(test_pl_filename, test_pl_metadata_filename, clf)


plot_predicted_vs_truth(test_predicted_class_band,
                        create_contour_classified_band(test_pl_filename,
                                                       test_ground_truth_filename))


# The predicted classes are close to the ground truth classes, but appear a little noisier and there are a few regions that are different.
# 
# First, let's look a a noisier region.
# 

plot_region_compare((slice(600,1000,None), slice(200,600,None)),
                    test_predicted_class_band,
                    create_contour_classified_band(test_pl_filename,
                                                   test_ground_truth_filename),
                    load_refl_bands(test_pl_filename, test_pl_metadata_filename))


# Ah! In this region, some of the 'noisier' appearance is actually due to the ground truth data being generalized a bit and missing some of the detail that the classified image is picking up. What is really cool is that the trees along the river are classified as 'non-crop' by the classifier, while the ground truth data classified the trees as 'crop' (possibly because the trees are shading the crop there).
# 
# However, there is some noise which is due to a misclassification of some of the crop pixels as 'non-crop' (lower-left, center-right regions). Hopefully segmentation will clean up these noisy regions.
# 
# Now let's check out a 'changed' region. This is a region that was marked as a non-crop region in the ground truth data, but is not predicted to be a crop by the KNN classifier. Lets zoom into rows 200-600, columns 400-800 to take a look.
# 

plot_region_compare((slice(200,600,None), slice(400,800,None)),
                    test_predicted_class_band,
                    create_contour_classified_band(test_pl_filename,
                                                   test_ground_truth_filename),
                    load_refl_bands(test_pl_filename, test_pl_metadata_filename))


# Wow! In the Planet scene RGB and NDVI, we can see that the crop shape has changed between when the ground truth data was collected and when the Planet scene was collected (2-year gap). The class predicted by the KNN classifier is more accurate than the ground truth class.
# 

# ### Classifier Accuracy Metric
# 
# Although the most informative way to determine the accuracy of the classification results is to compare the images of the predicted classes to the ground truth classes, this is subjective and can't be generalized. We will fall back on classic classification metrics and see if they match with our quantitative assessment.
# 
# scikit-learn provides a convenient function for calculating and reporting classification metrics, [`classification_report`](http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report). Using this, we will compare the classification metrics obtained with the train and the test datasets (in order).
# 

# train dataset
print(classification_report(to_y(create_contour_classified_band(train_pl_filename,
                                          train_ground_truth_filename)),
                            to_y(train_predicted_class_band),
                            target_names=['crop', 'non-crop']))


# test dataset
print(classification_report(to_y(create_contour_classified_band(test_pl_filename,
                                          test_ground_truth_filename)),
                            to_y(test_predicted_class_band),
                            target_names=['crop', 'non-crop']))


# The ultimate goal of the classification of pixels is accurate segmentation of crop regions. This likely will require that each region have a majority of correctly classified pixels. Therefore, both precision and recall are important and the f1-score is a good metric for predicting accuracy. 
# 
# The f1-score does drop between the train and test datasets, but considering that the classifier was trained on the 'train' dataset, the drop is to be expected. An f1-score of 0.77 is promising and is adequate to try segmentation schemes on the classified data.
# 

# ## Segmentation
# 
# We will perform segmentation on the test dataset classified by the trained KNN classifier.
# 

class_band = test_predicted_class_band.astype(np.uint8)

plt.figure(1, figsize=(8,8))
plt.imshow(class_band)
_ = plt.title('Test Predicted Classes')


# #### Denoising
# 
# We first clean up the image so that we don't end up with tiny segments caused by noise in the image. We use the median blur filter to remove speckle noise. This filter does a better job of preserving the road delineations between crops than the morphological open operation.
# 

def denoise(img, plot=False):
    def plot_img(img, title=None):
        if plot:
            figsize = 8; plt.figure(figsize=(figsize,figsize))
            plt.imshow(img)
            if title:
                plt.title(title)
        
    denoised = class_band.copy()
    plot_img(denoised, 'Original')
    
    denoised = cv2.medianBlur(denoised, 5)
    plot_img(denoised, 'Median Blur 5x5')
    
    return denoised

denoised = denoise(class_band, plot=True)


# #### Binary Segmentation
# 
# To convert the classified raster to vector features, we use the OpenCV [findContours](http://docs.opencv.org/trunk/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a) function. This function requires that the regions of interest have pixel values set to 1 with all other pixels have values set to 0.
# 

def to_crop_image(class_band):
    denoised = denoise(class_band)
    
    # crop pixels are 1, non-crop pixels are zero
    crop_image = np.zeros(denoised.shape, dtype=np.uint8)
    crop_image[(~class_band.mask) & (denoised == 0)] = 1
    return crop_image

crop_image = to_crop_image(class_band)
plt.figure(figsize=(8,8)); plt.imshow(crop_image)
_ = plt.title('Crop regions set to 1 (yellow)')


def get_crop_contours(img):
    """Identify contours that represent crop segments from a binary image.
    
    The crop pixel values must be 1, non-crop pixels zero.
    """
    # http://docs.opencv.org/trunk/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    def is_region(contour):
        num_points = contour.shape[0]
        return num_points >= 3

    return [c for c in contours
            if is_region(c)]

contours = get_crop_contours(crop_image)
print(len(contours))


figsize = 8; plt.figure(figsize=(figsize,figsize))
plt.imshow(draw_contours(bands_to_display(3*[np.ma.array(denoised, mask=class_band.mask)],
                                          alpha=False),
                         contours,
                         thickness=4))
_ = plt.title('Segmented Crop Outlines')

figsize = 8; plt.figure(figsize=(figsize,figsize))
plt.imshow(draw_contours(bands_to_display(get_rgb(load_refl_bands(test_pl_filename,
                                                                  test_pl_metadata_filename)),
                                          alpha=False),
                         contours,
                         thickness=-1))
_ = plt.title('Segmented Crop Regions')


# #### Create Crop Features
# 
# In this section, we use the source image spatial reference information to convert the OpenCV contours to georeferenced geojson features.
# 

# Utility functions: contours to features

def contour_to_polygon(contour, image_transform):
    """Convert an OpenCV contour to a rasterio Polygon.
    
    image_transform is typically obtained from `img.transform` where 
    img is obtained from `rasterio.open()
    
    :param contour: OpenCV contour
    :param rasterio.Affine image_transform: image transformation"""
    
    # get list of x and y coordinates from contour
    # contour: np.array(<list of points>, dtype=np.int32)
    # contour shape: (<number of points>, 1, 2)
    # point: (col, row)
    points_shape = contour.shape
    
    # get rid of the 1-element 2nd axis then convert 2d shape to a list
    cols_rows = contour.reshape((points_shape[0],points_shape[2])).tolist()
    cols, rows = zip(*cols_rows) # inverse of zip
    
    # convert rows/cols to x/y
    offset = 'ul' # OpenCV: contour point [0,0] is the upper-left corner of pixel 0,0
    points_x, points_y = rasterio.transform.xy(image_transform, rows, cols, offset=offset)
    
    # create geometry from series of points
    points_xy = zip(points_x, points_y)
    return Polygon(points_xy)


def project_features_from_srs(features, img_srs, dst_srs='epsg:4326'):
    """Project features from img_srs.
    
    If dst_srs is not specified, WGS84 (only geojson-supported crs) is assumed.
    
    :param features: list of geojson features to be projected
    :param str img_srs: source spatial reference system
    :param str dst_srs: destination spatial reference system
    """
    # define projection
    # from shapely [docs](http://toblerity.org/shapely/manual.html#shapely.ops.transform)
    proj_fcn = partial(
        pyproj.transform,
        pyproj.Proj(init=img_srs),
        pyproj.Proj(init=dst_srs))
    
    return [project_feature(f, proj_fcn) for f in features]


def polygon_to_feature(polygon):
    return {'type': 'Feature',
            'properties': {},
            'geometry': mapping(polygon)}    


def contours_to_features(contours, pl_filename):
    """Convert contours to geojson features.
    
    Features are simplified with a tolerance of 3 pixels then
    filtered to those with 100x100m2 area.
    """
    with rasterio.open(test_pl_filename) as img:
        img_srs = img.crs['init']
        img_transform = img.transform
        
    polygons = [contour_to_polygon(c, img_transform)
                for c in contours]

    tolerance = 3
    simplified_polygons = [p.simplify(tolerance, preserve_topology=True)
                           for p in polygons]

    def passes_filters(p):
        min_area = 50 * 50
        return p.area > min_area

    features = [polygon_to_feature(p)
                for p in simplified_polygons
                if passes_filters(p)]

    return project_features_from_srs(features, img_srs)

features = contours_to_features(contours, test_pl_filename)
print(len(features))


# #### Visualize Features on Map
# 

# Create crop feature layer
feature_collection = {
    "type": "FeatureCollection",
    "features": features,
    "properties": {"style":{
        'weight': 0,
        'fillColor': "blue",
        'fillOpacity': 1}}
}

feature_layer = ipyl.GeoJSON(data=feature_collection)


# Initialize map using parameters from above map
# and deleting map instance if it exists
try:
    del crop_map
except NameError:
    pass


zoom = 13
center = [38.30839, -121.55187] # lat/lon


# Create map, adding box drawing controls
# Reuse parameters if map already exists
try:
    center = crop_map.center
    zoom = crop_map.zoom
except NameError:
    pass

# Change tile layer to one that makes it easier to see crop features
# Layer selected using https://leaflet-extras.github.io/leaflet-providers/preview/
map_tiles = ipyl.TileLayer(url='http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png')
crop_map = ipyl.Map(
        center=center, 
        zoom=zoom,
        default_tiles = map_tiles
    )

crop_map.add_layer(feature_layer)  


# Display map
crop_map


# Alright! Those features look like they are georeferenced correctly and look reasonably 'crop-like'.
# 
# In this notebook, we have trained a KNN classifier to predict crop/non-crop pixels on one Planet image based on 2-year-old ground truth data, and we used that classifier to predict crop/non-crop pixels in another Planet image. We then segmented the classified image and created georeferenced geojson features that outline the predicted crops. It's been quite a journey and the results are quite promising!
# 




# # KNN Parameter Tuning
# 
# In [`Segmentation: KNN`](segment-knn.ipynb), we perform KNN classification of pixels as crop or non-crop. One parameter in the KNN classifier is the number of neighbors (the K in KNN). To determine what value this parameter should be, we perform cross-validation and pick the k that corresponds to the highest accuracy score. In this notebook, we demonstrate that cross-validation, using the training data X (values) and y (classifications) that was generated in `Segmentation: KNN`. The k value is then fed back into `Segmentation: KNN` to create the KNN Classifier that is used to predict pixel crop/non-crop designation.
# 
# In this notebook, we find that increasing the number of neighbors from 3 to 9 increases accuracy only marginally, while it also increases run time. Therefore, we will use the smallest number of neighbors: 3.
# 

from __future__ import print_function

import os

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier as KNN


# First we load the data that was saved in `Segmentation: KNN`
# 

# Load data
def load_cross_val_data(datafile):
    npzfile = np.load(datafile)
    X = npzfile['X']
    y = npzfile['y']
    return X,y

datafile = os.path.join('data', 'knn_cross_val', 'xy_file.npz')
X, y = load_cross_val_data(datafile)


# Next, we perform a grid search over the number of neighbors, looking for the value that corresponds to the highest accuracy.
# 

tuned_parameters = {'n_neighbors': range(3,11,2)}

clf = GridSearchCV(KNN(n_neighbors=3),
                   tuned_parameters,
                   cv=3,
                   verbose=10)
clf.fit(X, y)

print("Best parameters set found on development set:\n")
print(clf.best_params_)

print("Grid scores on development set:\n")

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
res_params = clf.cv_results_['params']
for mean, std, params in zip(means, stds, res_params):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


# It turns out that increasing the number of neighbors from 3 to 9 increases accuracy only marginally, while it also increases run time. Therefore, we will use the smallest number of neighbors: 3.
# 




# # Calculate Coverage
# 
# You have a large region of interest. You need to identify an AOI for your study. One of the inputs to that decision is the coverage within the region. This notebook will walk you through that process.
# 
# In this notebook, we create the coverage map for PS Orthotiles collected in 2017 through August for the state of Iowa. The coverage calculation is performed in WGS84 because it covers a larger area than a single UTM zone.
# 
# Ideas for improvements:
# - investigate projection
# 

# Notebook dependencies
from __future__ import print_function

import datetime
import copy
# from functools import partial
import os

from IPython.display import display # , Image
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from planet import api
from planet.api import filters
# import pyproj
import rasterio
from rasterio import features as rfeatures
from shapely import geometry as sgeom
# import shapely.ops

get_ipython().magic('matplotlib inline')


# ## Define AOI
# 
# Define the AOI as a geojson polygon. This can be done at [geojson.io](http://geojson.io). If you use geojson.io, only copy the single aoi feature, not the entire feature collection.
# 

iowa = {"geometry": {"type": "Polygon", "coordinates": [[[-91.163064, 42.986781], [-91.14556, 42.90798], [-91.143375, 42.90467], [-91.117411, 42.895837], [-91.100565, 42.883078], [-91.077643, 42.803798], [-91.069549, 42.769628], [-91.064896, 42.757272], [-91.053733, 42.738238], [-90.976314, 42.695996], [-90.949213, 42.685573], [-90.896961, 42.674407], [-90.84391, 42.663071], [-90.769495, 42.651443], [-90.720209, 42.640758], [-90.709204, 42.636078], [-90.702671, 42.630756], [-90.645627, 42.5441], [-90.636727, 42.518702], [-90.636927, 42.513202], [-90.640927, 42.508302], [-90.655927, 42.491703], [-90.656527, 42.489203], [-90.656327, 42.483603], [-90.654027, 42.478503], [-90.646727, 42.471904], [-90.624328, 42.458904], [-90.559451, 42.430695], [-90.477279, 42.383794], [-90.474834, 42.381473], [-90.419027, 42.328505], [-90.416535, 42.325109], [-90.4162, 42.321314], [-90.424326, 42.293326], [-90.430735, 42.284211], [-90.430884, 42.27823], [-90.419326, 42.254467], [-90.391108, 42.225473], [-90.375129, 42.214811], [-90.356964, 42.205445], [-90.349162, 42.204277], [-90.316269, 42.1936], [-90.306531, 42.190439], [-90.234919, 42.165431], [-90.211328, 42.15401], [-90.167533, 42.122475], [-90.162225, 42.11488], [-90.161119, 42.104404], [-90.163405, 42.087613], [-90.168358, 42.075779], [-90.166495, 42.054543], [-90.164537, 42.045007], [-90.154221, 42.033073], [-90.150916, 42.02944], [-90.141167, 42.008931], [-90.140613, 41.995999], [-90.1516, 41.931002], [-90.152104, 41.928947], [-90.181973, 41.80707], [-90.187969, 41.803163], [-90.20844, 41.797176], [-90.222263, 41.793133], [-90.242747, 41.783767], [-90.278633, 41.767358], [-90.302782, 41.750031], [-90.309826, 41.743321], [-90.31522, 41.734264], [-90.317041, 41.729104], [-90.334525, 41.679559], [-90.343162, 41.648141], [-90.34165, 41.621484], [-90.39793, 41.572233], [-90.461432, 41.523533], [-90.474332, 41.519733], [-90.499475, 41.518055], [-90.605937, 41.494232], [-90.655839, 41.462132], [-90.737537, 41.450127], [-90.771672, 41.450761], [-90.786282, 41.452888], [-90.847458, 41.455019], [-90.989976, 41.431962], [-91.027787, 41.423603], [-91.039872, 41.418523], [-91.047819, 41.4109], [-91.078682, 41.336089], [-91.079657, 41.333727], [-91.114186, 41.250029], [-91.113648, 41.241401], [-91.049808, 41.178033], [-91.019036, 41.16491], [-91.005503, 41.165622], [-90.997906, 41.162564], [-90.989663, 41.155716], [-90.946627, 41.096632], [-90.949383, 41.072711], [-90.949383, 41.07271], [-90.948523, 41.070248], [-90.945549, 41.06173], [-90.942253, 41.034702], [-90.952233, 40.954047], [-90.962916, 40.924957], [-90.968995, 40.919127], [-90.9985, 40.90812], [-91.009536, 40.900565], [-91.092993, 40.821079], [-91.097553, 40.808433], [-91.097031, 40.802471], [-91.094728, 40.797833], [-91.11194, 40.697018], [-91.112258, 40.696218], [-91.122421, 40.670675], [-91.138055, 40.660893], [-91.185415, 40.638052], [-91.18698, 40.637297], [-91.197906, 40.636107], [-91.218437, 40.638437], [-91.253074, 40.637962], [-91.306524, 40.626231], [-91.339719, 40.613488], [-91.348733, 40.609695], [-91.359873, 40.601805], [-91.405241, 40.554641], [-91.406851, 40.547557], [-91.404125, 40.539127], [-91.384531, 40.530948], [-91.369059, 40.512532], [-91.364211, 40.500043], [-91.36391, 40.490122], [-91.372554, 40.4012], [-91.375746, 40.391879], [-91.38836, 40.384929], [-91.419422, 40.378264], [-91.484507, 40.3839], [-91.490977, 40.393484], [-91.524612, 40.410765], [-91.619486, 40.507134], [-91.622362, 40.514362], [-91.618028, 40.53403], [-91.620071, 40.540817], [-91.696359, 40.588148], [-91.716769, 40.59853], [-91.729115, 40.61364], [-91.785916, 40.611488], [-91.795374, 40.611101], [-91.800133, 40.610953], [-91.813968, 40.610526], [-91.824826, 40.610191], [-91.832481, 40.609797], [-91.868401, 40.608059], [-91.943114, 40.605913], [-91.970988, 40.605112], [-91.998683, 40.604433], [-92.029649, 40.603713], [-92.067904, 40.602648], [-92.069521, 40.602772], [-92.082339, 40.602176], [-92.0832, 40.602244], [-92.092875, 40.602082], [-92.096387, 40.60183], [-92.17978, 40.600529], [-92.196162, 40.600069], [-92.201669, 40.59998], [-92.217603, 40.599832], [-92.236484, 40.599531], [-92.298754, 40.598469], [-92.331205, 40.597805], [-92.331445, 40.597714], [-92.350776, 40.597274], [-92.350807, 40.597273], [-92.379691, 40.596509], [-92.453745, 40.595288], [-92.461609, 40.595355], [-92.481692, 40.594941], [-92.482394, 40.594894], [-92.484588, 40.594924], [-92.580278, 40.592151], [-92.637898, 40.590853], [-92.639223, 40.590825], [-92.686693, 40.589809], [-92.689854, 40.589884], [-92.714598, 40.589564], [-92.742232, 40.589207], [-92.757407, 40.588908], [-92.828061, 40.588593], [-92.827992, 40.588515], [-92.835074, 40.588484], [-92.857391, 40.58836], [-92.863034, 40.588175], [-92.879178, 40.588341], [-92.889796, 40.588039], [-92.903544, 40.58786], [-92.941595, 40.587743], [-92.957747, 40.58743], [-93.085517, 40.584403], [-93.097296, 40.584014], [-93.098507, 40.583973], [-93.260612, 40.580797], [-93.317605, 40.580671], [-93.345442, 40.580514], [-93.374386, 40.580334], [-93.441767, 40.579916], [-93.465297, 40.580164], [-93.466887, 40.580072], [-93.524124, 40.580481], [-93.527607, 40.580436], [-93.528177, 40.580367], [-93.548284, 40.580417], [-93.553986, 40.580303], [-93.556899, 40.580235], [-93.558938, 40.580189], [-93.560798, 40.580304], [-93.56524, 40.580143], [-93.56581, 40.580075], [-93.566189, 40.580117], [-93.597352, 40.579496], [-93.656211, 40.578352], [-93.659272, 40.57833], [-93.661913, 40.578354], [-93.668845, 40.578241], [-93.677099, 40.578127], [-93.690333, 40.577875], [-93.722443, 40.577641], [-93.728355, 40.577547], [-93.737259, 40.577542], [-93.742759, 40.577518], [-93.750223, 40.57772], [-93.770231, 40.577615], [-93.774344, 40.577584], [-93.815485, 40.577278], [-93.818725, 40.577086], [-93.84093, 40.576791], [-93.853656, 40.576606], [-93.898327, 40.576011], [-93.899317, 40.575942], [-93.900877, 40.575874], [-93.913961, 40.575672], [-93.935687, 40.57533], [-93.936317, 40.575284], [-93.937097, 40.575421], [-93.938627, 40.575284], [-93.939857, 40.575192], [-93.963863, 40.574754], [-93.976766, 40.574635], [-94.015492, 40.573914], [-94.034134, 40.573585], [-94.080223, 40.572899], [-94.080463, 40.572899], [-94.089194, 40.572806], [-94.091085, 40.572897], [-94.23224, 40.571907], [-94.28735, 40.571521], [-94.294813, 40.571341], [-94.310724, 40.571524], [-94.324765, 40.571477], [-94.336556, 40.571475], [-94.336706, 40.571452], [-94.358307, 40.571363], [-94.429725, 40.571041], [-94.460088, 40.570947], [-94.470648, 40.57083], [-94.471213, 40.570825], [-94.48928, 40.570707], [-94.533878, 40.570739], [-94.537058, 40.570763], [-94.538318, 40.570763], [-94.541828, 40.570809], [-94.542154, 40.570809], [-94.594001, 40.570966], [-94.632032, 40.571186], [-94.632035, 40.571186], [-94.682601, 40.571787], [-94.714925, 40.572201], [-94.716665, 40.572201], [-94.773988, 40.572977], [-94.811188, 40.573532], [-94.819978, 40.573714], [-94.823758, 40.573942], [-94.896801, 40.574738], [-94.901451, 40.574877], [-94.914896, 40.575068], [-94.955134, 40.575669], [-94.966491, 40.575839], [-95.068921, 40.57688], [-95.079742, 40.577007], [-95.097607, 40.577168], [-95.107213, 40.577116], [-95.110303, 40.57716], [-95.110663, 40.577206], [-95.112222, 40.577228], [-95.120829, 40.577413], [-95.154499, 40.57786], [-95.164058, 40.578017], [-95.202264, 40.578528], [-95.211408, 40.57865], [-95.21159, 40.578654], [-95.212715, 40.578679], [-95.213327, 40.578689], [-95.217455, 40.578759], [-95.218783, 40.578781], [-95.221525, 40.578827], [-95.335588, 40.579871], [-95.357802, 40.5801], [-95.373893, 40.580501], [-95.373923, 40.580501], [-95.415406, 40.581014], [-95.469319, 40.58154], [-95.525392, 40.58209], [-95.526682, 40.582136], [-95.533182, 40.582249], [-95.554959, 40.582629], [-95.574046, 40.582963], [-95.611069, 40.583495], [-95.64184, 40.584234], [-95.687442, 40.58438], [-95.6875, 40.584381], [-95.746443, 40.584935], [-95.765645, 40.585208], [-95.753148, 40.59284], [-95.750053, 40.597052], [-95.748626, 40.603355], [-95.776251, 40.647463], [-95.786568, 40.657253], [-95.795489, 40.662384], [-95.822913, 40.66724], [-95.842801, 40.677496], [-95.883178, 40.717579], [-95.888907, 40.731855], [-95.88669, 40.742101], [-95.881529, 40.750611], [-95.872281, 40.758349], [-95.861695, 40.762871], [-95.854172, 40.784012], [-95.821193, 40.876682], [-95.823123, 40.900924], [-95.829074, 40.975688], [-95.835434, 40.984184], [-95.867286, 41.001599], [-95.867246, 41.043671], [-95.866289, 41.051731], [-95.853396, 41.16028], [-95.852788, 41.165398], [-95.91459, 41.185098], [-95.92319, 41.190998], [-95.923219, 41.191046], [-95.92599, 41.195698], [-95.927491, 41.202198], [-95.924891, 41.211198], [-95.90249, 41.273398], [-95.91379, 41.320197], [-95.92569, 41.322197], [-95.939291, 41.328897], [-95.953091, 41.339896], [-95.956691, 41.345496], [-95.956791, 41.349196], [-95.93831, 41.392162], [-95.937346, 41.394403], [-95.930705, 41.433894], [-95.981319, 41.506837], [-95.994784, 41.526242], [-96.030593, 41.527292], [-96.036603, 41.509047], [-96.040701, 41.507076], [-96.046707, 41.507085], [-96.055096, 41.509577], [-96.089714, 41.531778], [-96.09409, 41.539265], [-96.118105, 41.613495], [-96.116233, 41.621574], [-96.097728, 41.639633], [-96.095046, 41.647365], [-96.095415, 41.652736], [-96.099837, 41.66103], [-96.121726, 41.68274], [-96.096795, 41.698681], [-96.077088, 41.715403], [-96.064537, 41.793002], [-96.06577, 41.798174], [-96.071007, 41.804639], [-96.077646, 41.808804], [-96.086407, 41.81138], [-96.110907, 41.830818], [-96.139554, 41.86583], [-96.144483, 41.871854], [-96.161756, 41.90182], [-96.161988, 41.905553], [-96.159098, 41.910057], [-96.142265, 41.915379], [-96.136743, 41.920826], [-96.129186, 41.965136], [-96.129505, 41.971673], [-96.22173, 42.026205], [-96.251714, 42.040472], [-96.272877, 42.047238], [-96.279079, 42.074026], [-96.307421, 42.130707], [-96.344121, 42.162091], [-96.349688, 42.172043], [-96.35987, 42.210545], [-96.356666, 42.215077], [-96.356591, 42.215182], [-96.336323, 42.218922], [-96.323723, 42.229887], [-96.322868, 42.233637], [-96.328905, 42.254734], [-96.348814, 42.282024], [-96.375307, 42.318339], [-96.384169, 42.325874], [-96.407998, 42.337408], [-96.413895, 42.343393], [-96.417786, 42.351449], [-96.415509, 42.400294], [-96.413609, 42.407894], [-96.387608, 42.432494], [-96.380707, 42.446394], [-96.381307, 42.461694], [-96.385407, 42.473094], [-96.396107, 42.484095], [-96.409408, 42.487595], [-96.443408, 42.489495], [-96.466253, 42.497702], [-96.476947, 42.508677], [-96.481308, 42.516556], [-96.479909, 42.524195], [-96.477709, 42.535595], [-96.476952, 42.556079], [-96.479685, 42.561238], [-96.516338, 42.630435], [-96.542366, 42.660736], [-96.575299, 42.682665], [-96.601989, 42.697429], [-96.60614, 42.694661], [-96.610975, 42.694751], [-96.630617, 42.70588], [-96.639704, 42.737071], [-96.633168, 42.768325], [-96.632142, 42.770863], [-96.577813, 42.828102], [-96.563058, 42.831051], [-96.552092, 42.836057], [-96.549513, 42.839143], [-96.54146, 42.857682], [-96.523264, 42.909059], [-96.510749, 42.944397], [-96.509479, 42.971122], [-96.513111, 43.02788], [-96.466017, 43.062235], [-96.455107, 43.083366], [-96.439335, 43.113916], [-96.436589, 43.120842], [-96.475571, 43.221054], [-96.485264, 43.224183], [-96.557126, 43.224192], [-96.572489, 43.249178], [-96.584124, 43.268101], [-96.586317, 43.274319], [-96.56911, 43.295535], [-96.551929, 43.292974], [-96.530392, 43.300034], [-96.525564, 43.312467], [-96.521264, 43.374978], [-96.521697, 43.386897], [-96.524044, 43.394762], [-96.529152, 43.397735], [-96.531159, 43.39561], [-96.53746, 43.395246], [-96.557586, 43.406792], [-96.594254, 43.434153], [-96.60286, 43.450907], [-96.600039, 43.45708], [-96.58407, 43.468856], [-96.587151, 43.484697], [-96.598928, 43.500457], [-96.591213, 43.500514], [-96.453049, 43.500415], [-96.351059, 43.500333], [-96.332062, 43.500415], [-96.208814, 43.500391], [-96.198766, 43.500312], [-96.198484, 43.500335], [-96.053163, 43.500176], [-95.861152, 43.499966], [-95.860946, 43.499966], [-95.834421, 43.499966], [-95.821277, 43.499965], [-95.741569, 43.499891], [-95.740813, 43.499894], [-95.514774, 43.499865], [-95.486803, 43.500246], [-95.486737, 43.500274], [-95.475065, 43.500335], [-95.454706, 43.500563], [-95.454706, 43.500648], [-95.454433, 43.500644], [-95.434293, 43.50036], [-95.434199, 43.500314], [-95.387851, 43.50024], [-95.387812, 43.50024], [-95.387787, 43.50024], [-95.375269, 43.500322], [-95.374737, 43.500314], [-95.250969, 43.500464], [-95.250762, 43.500406], [-95.214938, 43.500885], [-95.180423, 43.500774], [-95.167891, 43.500885], [-95.167294, 43.500771], [-95.122633, 43.500755], [-95.114874, 43.500667], [-95.054289, 43.50086], [-95.053504, 43.500769], [-95.034, 43.500811], [-95.014245, 43.500872], [-94.99446, 43.500523], [-94.974359, 43.500508], [-94.954477, 43.500467], [-94.934625, 43.50049], [-94.914955, 43.50045], [-94.914905, 43.50045], [-94.914634, 43.50045], [-94.914523, 43.50045], [-94.887291, 43.500502], [-94.874235, 43.500557], [-94.872725, 43.500564], [-94.860192, 43.500546], [-94.857867, 43.500615], [-94.854555, 43.500614], [-94.615916, 43.500544], [-94.565665, 43.50033], [-94.560838, 43.500377], [-94.47042, 43.50034], [-94.447048, 43.500639], [-94.442848, 43.500583], [-94.442835, 43.500583], [-94.390597, 43.500469], [-94.377466, 43.500379], [-94.247965, 43.500333], [-94.10988, 43.500283], [-94.108068, 43.5003], [-94.094339, 43.500302], [-94.092894, 43.500302], [-93.970762, 43.499605], [-93.97076, 43.499605], [-93.795793, 43.49952], [-93.794285, 43.499542], [-93.716217, 43.499563], [-93.708771, 43.499564], [-93.704916, 43.499568], [-93.699345, 43.499576], [-93.648533, 43.499559], [-93.617131, 43.499548], [-93.576728, 43.49952], [-93.558631, 43.499521], [-93.532178, 43.499472], [-93.528482, 43.499471], [-93.497405, 43.499456], [-93.49735, 43.499456], [-93.488261, 43.499417], [-93.482009, 43.499482], [-93.472804, 43.4994], [-93.468563, 43.499473], [-93.428509, 43.499478], [-93.399035, 43.499485], [-93.2718, 43.499356], [-93.228861, 43.499567], [-93.049192, 43.499571], [-93.024429, 43.499572], [-93.024348, 43.499572], [-93.007871, 43.499604], [-92.870277, 43.499548], [-92.790317, 43.499567], [-92.752088, 43.500084], [-92.707312, 43.500069], [-92.692786, 43.500063], [-92.689033, 43.500062], [-92.67258, 43.500055], [-92.653318, 43.50005], [-92.649194, 43.500049], [-92.553161, 43.5003], [-92.553128, 43.5003], [-92.464505, 43.500345], [-92.448948, 43.50042], [-92.408832, 43.500614], [-92.40613, 43.500476], [-92.388298, 43.500483], [-92.368908, 43.500454], [-92.279084, 43.500436], [-92.277425, 43.500466], [-92.198788, 43.500527], [-92.178863, 43.500713], [-92.103886, 43.500735], [-92.08997, 43.500684], [-92.079954, 43.500647], [-92.079802, 43.500647], [-91.949879, 43.500485], [-91.941837, 43.500554], [-91.824848, 43.500684], [-91.807156, 43.500648], [-91.804925, 43.500716], [-91.77929, 43.500803], [-91.777688, 43.500711], [-91.761414, 43.500637], [-91.738446, 43.500525], [-91.736558, 43.500561], [-91.73333, 43.500623], [-91.730359, 43.50068], [-91.730217, 43.50068], [-91.700749, 43.500581], [-91.670872, 43.500513], [-91.658401, 43.500533], [-91.651396, 43.500454], [-91.644924, 43.500529], [-91.639772, 43.500573], [-91.635626, 43.500463], [-91.634495, 43.500439], [-91.634244, 43.500479], [-91.625611, 43.500727], [-91.620785, 43.500677], [-91.617407, 43.500687], [-91.616895, 43.500663], [-91.615293, 43.50055], [-91.610895, 43.50053], [-91.610832, 43.50053], [-91.591073, 43.500536], [-91.551021, 43.500539], [-91.54122, 43.500515], [-91.533806, 43.50056], [-91.491042, 43.50069], [-91.465063, 43.500608], [-91.461403, 43.500642], [-91.445932, 43.500588], [-91.441786, 43.500438], [-91.37695, 43.500482], [-91.371608, 43.500945], [-91.369325, 43.500827], [-91.217706, 43.50055], [-91.20555, 43.422949], [-91.210233, 43.372064], [-91.107237, 43.313645], [-91.085652, 43.29187], [-91.057918, 43.255366], [-91.062562, 43.243165], [-91.1462, 43.152405], [-91.1562, 43.142945], [-91.175253, 43.134665], [-91.178251, 43.124982], [-91.177222, 43.080247], [-91.178087, 43.062044], [-91.175167, 43.041267], [-91.163064, 42.986781]]]}, "type": "Feature", "properties": {"CENSUSAREA": 55857.13, "STATE": "19", "LSAD": "", "NAME": "Iowa", "GEO_ID": "0400000US19"}}


# this notebook uses rasterio Shapes for processing, so lets convert that geojson to a shape
aoi_shape = sgeom.shape(iowa['geometry'])
aoi_shape


# ## Build Request
# 
# Build the Planet API Filter request.
# 
# Customize this code for your own purposes
# 

DATE_START = datetime.datetime(year=2017,month=1,day=1)
DATE_END = datetime.datetime(year=2017,month=9,day=1)

def build_request(aoi_shape, date_start=DATE_START, date_end=DATE_END, addl_filters=None):
    base_filters = [
        filters.geom_filter(sgeom.mapping(aoi_shape)),
        filters.range_filter('cloud_cover', lt=.1),
        filters.date_range('acquired', gt=date_start),
        filters.date_range('acquired', lt=date_end),
    ]
    
    if addl_filters is not None:
        base_filters += addl_filters

    query = filters.and_filter(*base_filters)
    
    item_types = ['PSOrthoTile']
    return filters.build_search_request(query, item_types)

request = build_request(aoi_shape)
# print(request)


# ## Set Coverage Grid Dimensions
# 
# Set the grid dimensions according to the AOI shape and the resolution of interest
# 

dimensions = (3000, 4000)


# ## Search Planet API
# 
# The client is how we interact with the planet api. It is created with the user-specific api key, which is pulled from $PL_API_KEY environment variable.
# 

def get_api_key():
    return os.environ['PL_API_KEY']


# quick check that key is defined
assert get_api_key(), "PL_API_KEY not defined."


import json


def create_client():
    return api.ClientV1(api_key=get_api_key())


def search_pl_api(request, limit=500):
    client = create_client()
    result = client.quick_search(request)
    
    # note that this returns a generator
    return result.items_iter(limit=limit)

item = next(search_pl_api(build_request(aoi_shape), limit=1))
print(json.dumps(item['properties']))


# ## Calculate Coverage
# 
# First query the planet api for the items that match the request defined above, then calculate the overlap between each item and the aoi. Finally, convert each overlap to a grid using [`rasterio.rasterize`](https://mapbox.github.io/rasterio/topics/features.html#burning-shapes-into-a-raster), accumulate coverage over the overlap grids, and display the coverage grid.
# 

def calculate_overlap(item, aoi_shape):
    footprint_shape = sgeom.shape(item['geometry'])
    return aoi_shape.intersection(footprint_shape)

def calculate_overlaps(items, aoi_shape):
    item_num = 0
    overlap_num = 0
    for i in items:
        item_num += 1
        overlap = calculate_overlap(i, aoi_shape)
        if not overlap.is_empty:
            overlap_num += 1
            yield overlap
    print('{} overlaps from {} items'.format(overlap_num, item_num))


def calculate_coverage(overlaps, dimensions, bounds):
    
    # get dimensions of coverage raster
    mminx, mminy, mmaxx, mmaxy = bounds

    y_count, x_count = dimensions
    
    # determine pixel width and height for transform
    width = (mmaxx - mminx) / x_count
    height = (mminy - mmaxy) / y_count # should be negative

    # Affine(a, b, c, d, e, f) where:
    # a = width of a pixel
    # b = row rotation (typically zero)
    # c = x-coordinate of the upper-left corner of the upper-left pixel
    # d = column rotation (typically zero)
    # e = height of a pixel (typically negative)
    # f = y-coordinate of the of the upper-left corner of the upper-left pixel
    # ref: http://www.perrygeo.com/python-affine-transforms.html
    transform = rasterio.Affine(width, 0, mminx, 0, height, mmaxy)
    
    coverage = np.zeros(dimensions, dtype=np.uint16)
    for overlap in overlaps:
        if not overlap.is_empty:
            # rasterize overlap vector, transforming to coverage raster
            # pixels inside overlap have a value of 1, others have a value of 0
            overlap_raster = rfeatures.rasterize(
                    [sgeom.mapping(overlap)],
                    fill=0,
                    default_value=1,
                    out_shape=dimensions,
                    transform=transform)
            
            # add overlap raster to coverage raster
            coverage += overlap_raster
    return coverage


items = search_pl_api(request = build_request(aoi_shape),
                      limit=10000)
overlaps = calculate_overlaps(items, aoi_shape)

# cache coverage calculation because it takes a little while to create
coverage = calculate_coverage(overlaps, dimensions, aoi_shape.bounds)


from scipy import stats as sstats


import matplotlib.colors as colors

def plot_coverage(coverage):
    fig, ax = plt.subplots(figsize=(15,10))
    
    # ref: https://matplotlib.org/users/colormapnorms.html
    pcm = ax.imshow(coverage,
                       interpolation='nearest',
                       norm=colors.LogNorm(vmin=max(1, coverage.min()), # avoid divide by zero
                                           vmax=coverage.max()),
                       cmap=cm.viridis)
    fig.colorbar(pcm, ax=ax, extend='max')
    fig.show()

    ax.set_title('Coverage\n(median: {})'.format(int(np.median(coverage))))
    ax.axis('off')

plot_coverage(coverage)


# Even when we limit the query to 10,000 scenes (there are more scenes than that, but it takes quite some time to process that many scenes), there are areas in Iowa that have 10-100 scenes of coverage. Pretty awesome!
# 




# ## Creating a composite image from multiple PlanetScope scenes
# 

# In this guide, you'll learn how to create a composite image (or mosaic) from multiple PlanetScope scenes that cover an area of interest (AOI). You'll need [GDAL (Geospatial Data Abstraction Library)](http://www.gdal.org/) and its python bindings installed to run the commands below.
# 

# First, let's use [Planet Explorer](https://www.planet.com/explorer/) to travel to stunning Yosemite National Park. You can see below that I've drawn an area of interest around [Mount Dana](https://en.wikipedia.org/wiki/Mount_Dana) on the eastern border of Yosemite. I want an image that depicts the mountain on a clear summer day, so I've narrowed my data search in Planet Explorer to scenes with less than 5% cloud cover, captured in July and August 2016. 
# 

# ![Mount Dana in Planet Explorer](images/pe-mtdana.gif)

# As you can see in the animated gif above, my search yielded a set of three PlanetScope scenes, all taken on August 20, 2016. Together these scenes cover 100% of my area of interest. As I roll over each item in Planet Explorer, I can see that the scenes' rectangular footprints extend far beyond Mount Dana. All three scenes overlap slightly, and one scene touches only a small section at the bottom of my AOI. Still, they look good to me, so I'm going to submit an order for the visual assets. 
# 

# After downloading, moving, and wrangling the data, I'm ready to create a composite image from the three scenes. First, though, I'll use `gdalinfo` to inspect the spatial metadata of the scenes. 
# 

get_ipython().system('gdalinfo data/175322.tif ')
get_ipython().system('gdalinfo data/175323.tif')
get_ipython().system('gdalinfo data/175325.tif')


# The three scenes have the same coordinate systems and the same number of bands, so we can go ahead and use the `gdal_merge.py` utility to stitch them together. In areas of overlap, the utility will copy over parts of the previous image in the list of input files with data from the next image. The `-v` flag in the command below allows us to see the output of the mosaicing operations as they are done. 
# 

get_ipython().system('gdal_merge.py -v data/175322.tif data/175323.tif data/175325.tif -o output/mtdana-merged.tif')


# We can see in the verbose output above that the mosaicing operation is fairly simple: the utility script is basically copying a range of pixels for each band in each scene over to the designated output file. We can use `gdalinfo` to inspect the metadata of the merged raster file we created.
# 

get_ipython().system('gdalinfo output/mtdana-merged.tif')


# The merged raster is a large GeoTiff file, so we're going use `gdal_translate`, another GDAL utility, to convert it to a PNG and set the output image to a percentage of the original. That will make it easier for us to view in this notebook. 
# 

get_ipython().system('gdal_translate -of "PNG" -outsize 10% 0% output/mtdana-merged.tif output/mtdana-merged.png')


# Now let's view the merged image.
# 

from IPython.display import Image
Image(filename="output/mtdana-merged.png")


# Success! Wait... this is definitely a composite image from our three PlanetScope scenes, but it's not really what we want. We'd much rather have a composite image that is cropped to the boundaries of our AOI. We can use `gdalwarp` to clip the raster to our area of interest (defined by a `geojson` file). 
# 

get_ipython().system('gdalwarp -of GTiff -cutline data/mt-dana-small.geojson -crop_to_cutline output/mtdana-merged.tif output/mtdana-cropped.tif')


# Again, we'll use `gdal_translate` to convert the GeoTiff to a smaller PNG so that it's easier to view the cropped image.
# 

get_ipython().system('gdal_translate -of "PNG" -outsize 10% 0% output/mtdana-cropped.tif output/mtdana-cropped.png')


from IPython.display import Image
Image(filename="output/mtdana-cropped.png")


# Success! A cropped, composite image of Mount Dana in Yosemite! [Who wants to go for a hike?](https://www.alltrails.com/trail/us/california/mount-dana-summit-trail)
# 

# # Crop Type Classification: CART L8 and PS
# 
# This notebook is a continuation of [Crop Type Classification: CART
# ](classify-cart.ipynb) in which we use Landsat 8 as well as the PS Orthotile to generate features for CART classification.
# 
# This notebook demonstrates the following:
# 1. Finding a Landsat 8 scene that overlaps a PS Orthotile
# 1. Resampling Landsat 8 bands to match a PS Orthotile
# 1. Loading, visualizing, and using bitwise logic to convert a Landsat 8 QA band to a mask
# 1. Training a CART classifier using a combination of features from the PS Orthotile and Landsat 8 scene
# 1. Quantifying the performance of the classifier on the training data (upper limit of performance on a new dataset)
# 
# To enable this notebook, a lot of functionality was copied from [Crop Type Classification: CART
# ](classify-cart.ipynb).
# 
# **NOTE** This notebook utilizes datasets that are downloaded and prepared in [Crop Type Classification: CART
# ](classify-cart.ipynb), so that notebook should be run first.
# 
# 
# ## Install Dependencies
# 

from collections import namedtuple, OrderedDict
import json
import os
from subprocess import check_output, STDOUT, CalledProcessError
import tempfile
from xml.dom import minidom

import matplotlib
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

get_ipython().magic('matplotlib inline')


# ## Identify Datasets
# 
# ### PS Orthotile
# 
# Files associated with PS Orthotile [210879_1558814_2016-07-25_0e16](https://api.planet.com/data/v1/item-types/PSOrthoTile/items/210879_1558814_2016-07-25_0e16/thumb) were downloaded as a part of [Crop Type Classification: CART
# ](classify-cart.ipynb)
# 

# define data file filenames and ensure they exist
train_folder = os.path.join('data', 'cart', '210879_1558814_2016-07-25_0e16')
print(train_folder)

train_files = {
    'scene': os.path.join(train_folder, '210879_1558814_2016-07-25_0e16_BGRN_Analytic.tif'),
    'metadata': os.path.join(train_folder, '210879_1558814_2016-07-25_0e16_BGRN_Analytic_metadata.xml'),
    'udm': os.path.join(train_folder, '210879_1558814_2016-07-25_0e16_BGRN_DN_udm.tif'),
}

for filename in train_files.values():
    print(filename)
    assert os.path.isfile(filename)


# ### Landsat 8 Scene
# 
# To find the Landsat 8 scene that corresponds to the PS Orthotile, we read the Orthotile footprint from the Orthotile metadata and save it as geojson for use in searching for Landsat 8 scenes in Planet Explorer. 
# 

# save Orthotile footprint as aoi geojson

def get_footprint_coords(metadata_filename):
    xmldoc = minidom.parse(metadata_filename)
    fp_node = xmldoc.getElementsByTagName('ps:Footprint')[0]

    # hone in on the footprint coordinates
    # the coordinates are always specified in WGS84, which is also the
    # geographic coordeinate system
    coords_node = fp_node.getElementsByTagName('gml:coordinates')[0]
    coords_str = coords_node.firstChild.data
    
    # coordinates entry is space-separated lat,lon
    coords = [[float(c) for c in cs.split(',')] for cs in coords_str.split(' ')]
    return coords

def coords_to_feature(coords):
    geom = {
        "type": "Polygon",
        "coordinates": [coords]
        }
    
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": geom}

    return feature
        
def write_geojson(feature, filename):
    with open(filename, "w") as f:
        f.write(json.dumps(feature))
    
    print(filename)
        
coords = get_footprint_coords(train_files['metadata'])
feat = coords_to_feature(coords)
print(json.dumps(feat))

# save aoi and add to list of files
aoi_filename = os.path.join(train_folder, 'aoi.geojson')
write_geojson(feat, aoi_filename)
train_files['aoi'] = aoi_filename


# #### Download Landsat Scene
# 

# In planet explorer, we upload the AOI as a filter geometry then filter the dates to July 1, 2016 to September 2, 2016. This query results in two Landsat 8 scenes, with [LC80260302016245LGN00](https://api.planet.com/data/v1/item-types/Landsat8L1G/items/LC80260302016245LGN00/thumb) standing out as the best scene to use.
# 

# uncomment below to learn more about Landsat8L1G LC80260302016245LGN00

# !planet data search --item-type Landsat8L1G --string-in id LC80260302016245LGN00

# uncomment below to download scene and supporting files to local folder

# !mkdir data/cart/LC80260302016245LGN00
# !planet data download --item-type Landsat8L1G \
#     --dest data/cart/LC80260302016245LGN00 \
#     --asset-type analytic_bqa,analytic_b2,analytic_b3,analytic_b4,analytic_b5,analytic_b6,analytic_b7 \
#     --string-in id LC80260302016245LGN00
# !ls -l --block-size=M data/cart/LC80260302016245LGN00


l8_filenames = {
    'qa': 'LC80260302016245LGN00_BQA.TIF',
    'b2': 'LC80260302016245LGN00_B2.TIF',
    'b3': 'LC80260302016245LGN00_B3.TIF',
    'b4': 'LC80260302016245LGN00_B4.TIF',
    'b5': 'LC80260302016245LGN00_B5.TIF',
#     'b6': 'LC80260302016245LGN00_B6.TIF',
#     'b7': 'LC80260302016245LGN00_B7.TIF'
}

src_l8_folder = 'data/cart/LC80260302016245LGN00'

def abs_path_filenames(folder, filenames):
    return dict([(k, os.path.join(folder, fn))
                 for k, fn in filenames.items()])

src_l8_files = abs_path_filenames(src_l8_folder, l8_filenames)
src_l8_files


# # Resample Landsat Scene to PS Orthotile
# 
# To stack the Landsat 8 and PS Orthothile bands, the pixels must be the same size and line up spatially. To accomplish this, we resample the Landsat 8 scene to the Orthotile.
# 

# Utility functions: crop, resample, and project an image

# These use gdalwarp. for a description of gdalwarp command line options, see:
# http://www.gdal.org/gdalwarp.html

# projection is not required for our application, where the Landsat
# scene and the PS Orthotile are projected to the same UTM Zone
# but this code is kept here in case that changes
# def gdalwarp_project_options(src_crs, dst_crs):
#     return ['-s_srs', src_crs.to_string(), '-t_srs', dst_crs.to_string()]

def gdalwarp_crop_options(bounds, crs):
    xmin, ymin, xmax, ymax = [str(b) for b in bounds]
    # -te xmin ymin xmax ymax
    return ['-te', xmin, ymin, xmax, ymax]

def gdalwarp_resample_options(width, height, technique='near'):
    # for technique options, see: http://www.gdal.org/gdalwarp.html
    return ['-ts', width, height, '-r', technique]

def gdalwarp(input_filename, output_filename, options):
    commands = _gdalwarp_commands(input_filename, output_filename, options)

    # print error if one is encountered
    # https://stackoverflow.com/questions/29580663/save-error-message-of-subprocess-command
    try:
        output = check_output(commands, stderr=STDOUT)
    except CalledProcessError as exc:
        print(exc.output)

def _gdalwarp_commands(input_filename, output_filename, options):
    commands = ['gdalwarp'] + options +                ['-overwrite',
                input_filename,
                output_filename]
    print(' '.join(commands))
    return commands

def _test():
    TEST_DST_SCENE = train_files['scene']
    TEST_SRC_SCENE = src_l8_files['qa']

    with rasterio.open(TEST_DST_SCENE, 'r') as dst:
        with rasterio.open(TEST_SRC_SCENE, 'r') as src:
#             print(gdalwarp_project_options(src.crs, dst.crs))
            print(gdalwarp_crop_options(dst.bounds, dst.crs))
            print(gdalwarp_resample_options(dst.width, dst.height))
# _test()


def prepare_l8_band(band_filename, dst_filename, out_filename, classified=False):
    '''Project, crop, and resample landsat 8 band to match dst_filename image.'''
    
    # use 'near' resampling method for classified (e.g. qa) band,
    # otherwise use 'cubic' method
    method = 'near' if classified else 'cubic'
    
    with rasterio.open(band_filename, 'r') as src:
        with rasterio.open(dst_filename, 'r') as dst:
            # project
            # proj_options = gdalwarp_project_options(src_crs, dst.crs)

            # crop
            crop_options = gdalwarp_crop_options(dst.bounds, dst.crs)

            # resample
            width, height = dst.shape
            resample_options = gdalwarp_resample_options(str(width), str(height), method)

            options = crop_options + resample_options
            
            # run gdalwarp
            gdalwarp(band_filename, out_filename, options)


def _test(delete=True):
    TEST_DST_SCENE = train_files['scene']
    TEST_SRC_SCENE = src_l8_files['qa']
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=delete, dir='.') as out_file:
        # create output
        prepare_l8_band(TEST_SRC_SCENE, TEST_DST_SCENE, out_file.name, classified=True)

        # check output
        with rasterio.open(TEST_DST_SCENE, 'r') as dst:
            with rasterio.open(out_file.name, 'r') as src:
                assert dst.crs == src.crs, '{} != {}'.format(src.crs, dst.crs)
                assert dst.bounds == src.bounds
                assert dst.shape == src.shape
# _test()


def prepare_l8_bands(src_files, dst_folder, ps_scene):
    dst_files = {}
    for name, filename in src_l8_files.items():
        # qa band is the only classified band
        classified = name=='qa'
        
        dst_file = os.path.join(dst_folder, os.path.basename(filename))
        prepare_l8_band(filename, ps_scene, dst_file, classified=classified)
        dst_files[name] = dst_file
    return dst_files

def _test():
    try:
        out_folder = tempfile.mkdtemp()
        dst_l8_files = prepare_l8_bands(src_l8_files, out_folder, train_files['scene'])
        print dst_l8_files
    finally:
        del out_folder
# _test()


train_l8_folder = 'data/cart/210879_1558814_2016-07-25_0e16/L8'

if not os.path.isdir(train_l8_folder):
    os.mkdir(train_l8_folder)
    print(train_l8_folder)


train_l8_files = prepare_l8_bands(src_l8_files, train_l8_folder, train_files['scene'])
train_l8_files


# ## Landsat 8 QA band
# 
# We use the Landsat 8 QA band to mask out any 'bad' (quality issue, cloud, etc) pixels. To accomplish this, first we create functionality for dealing with any generic classified band, then we load the QA band, visualize it, and convert it to a mask.
# 

# Utility functions: visualizing a classified band as an image

def plot_image(image, title, figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    image.imshow(ax)
    ax.set_title(title)
    ax.set_axis_off()
        
class ClassifiedImage(object):
    def __init__(self, band, labels=None):
        self.band = band
        self.labels = labels

    def imshow(self, ax, cmap='rainbow'):
        """Show classified band with colormap normalization and color legend.
        
        Alters ax in place.

        possible cmaps ref: https://matplotlib.org/examples/color/colormaps_reference.html
        """
        class_norm = _ClassNormalize(self.band)
        im = ax.imshow(self.band, cmap=cmap, norm=class_norm)

        try:
            # add class label legend
            # https://stackoverflow.com/questions/25482876
            # /how-to-add-legend-to-imshow-in-matplotlib
            color_mapping = class_norm.mapping
            colors = [im.cmap(color_mapping[k])
                      for k in self.labels.keys()]
            labels = self.labels.values()

            # https://matplotlib.org/users/legend_guide.html
            # tag: #creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
            patches = [mpatches.Patch(color=c, label=l)
                       for c,l in zip(colors, labels)]

            ax.legend(handles=patches, bbox_to_anchor=(1, 1),
                      loc='upper right', borderaxespad=0.)
        except AttributeError:
            # labels not specified
            pass


# https://matplotlib.org/users/colormapnorms.html#custom-normalization-two-linear-ranges
class _ClassNormalize(colors.Normalize):
    """Matplotlib colormap normalizer for a classified band.
    
    __init__ and __call__ are the minimum required methods.
    """
    def __init__(self, arry):
        # get unique unmasked values
        values = [v for v in np.unique(arry)
                  if not isinstance(v, np.ma.core.MaskedConstant)]

        color_ticks = np.array(range(len(values)), dtype=np.float) / (len(values) - 1)
        self._mapping = dict((v, ct)
                            for v, ct in zip(values, color_ticks))
        
        # Initialize base Normalize instance
        vmin = 0
        vmax = 1
        clip = False
        colors.Normalize.__init__(self, vmin, vmax, clip)
    
    def __call__(self, arry, clip=None):
        # round array back to ints for logical comparison
        arry = np.around(arry)
        new_arry = arry.copy()
        for k, v in self._mapping.items():
            new_arry[arry==k] = v
        return new_arry
    
    @property
    def mapping(self):
        return self._mapping

def _test():
#     classified_band = np.array(range(4)).reshape((2,2))
    classified_band = np.array([0, 1, 2, 28, 30, 64, 66, 92, 94], dtype=np.uint8).reshape((3,3))
    print(classified_band)
    labels = OrderedDict((v, str(v)) for v in np.unique(classified_band))
    classified_image = ClassifiedImage(band=classified_band, labels=labels)
    plot_image(classified_image, title='Test', figsize=(4,4))
# _test()


# Functionality specific to the QA Band
# 

# Utility function for working with large images and memory limitations

def _read_window(filename, window):
    with rasterio.open(filename, 'r') as src:
        return src.read(window=window)

def decimated(arry, num=8):
    return arry[::num, ::num].copy()


def load_band(band_filename, window=None):
    return _read_window(band_filename, window)[0,...]

def get_qa_labels(binary_band):
    return OrderedDict((v, '{0:b}'.format(v).zfill(16))
                       for v in np.unique(binary_band))

def _test():
    qa = decimated(load_band(train_l8_files['qa']))
    qa_labels = get_qa_labels(qa)
    qa_classified_band = ClassifiedImage(band=qa, labels=qa_labels)
    plot_image(qa_classified_band, title='QA Band', figsize=(7,7))
_test()


def qa_to_mask(qa_array):
    """Generate mask from L8 QA band.
    
    Pre-Collection:
    The description for the pre-collection QA band is no longer hosted by USGS.
    We pull it from plcompositor, information encoded in the bits in the QA is from:
    https://github.com/planetlabs/plcompositor/blob/master/src/landsat8cloudquality.cpp#L28
    
    For the single bits (0, 1, 2, and 3):
    0 = No, this condition does not exist
    1 = Yes, this condition exists.
    
    The double bits (4-5, 6-7, 8-9, 10-11, 12-13, and 14-15)
    represent levels of confidence that a condition exists:
    00 = Algorithm did not determine the status of this condition
    01 = Algorithm has low confidence that this condition exists 
         (0-33 percent confidence)
    10 = Algorithm has medium confidence that this condition exists 
         (34-66 percent confidence)
    11 = Algorithm has high confidence that this condition exists 
         (67-100 percent confidence).

     Mask    Meaning
    0x0001 - Designated Fill0
    0x0002 - Dropped Frame
    0x0004 - Terrain Occlusion
    0x0008 - Reserved
    0x0030 - Water Confidence
    0x00c0 - Reserved for cloud shadow
    0x0300 - Vegitation confidence
    0x0c00 - Show/ice Confidence
    0x3000 - Cirrus Confidence
    0xc000 - Cloud Confidence
    
    Collection 1:
    
    The description for the information encoded in the bits in the QA is from:
    https://landsat.usgs.gov/collectionqualityband
    
    Bit 0: designated fill
    Bit 1: terrain occlusion
    Bits 2/3: radiometric saturation
    Bit 4: cloud
    Bits 5/6: cloud confidence
    Bits 7/8: cloud shadow confidence
    Bits 9/10: snow/ice confidence
    Bits 11/12: cirr
    """
    # check for absolute or >= med confidence of any condition
    test_bits = int('1010101000001111',2)
    bit_matches = qa_array & test_bits # bit-wise logical AND operator
    return bit_matches != 0 # mask any pixels that match test bits

def _test():
    mask = qa_to_mask(load_band(train_l8_files['qa']))
    print'{}/{} ({:0.1f}%) masked'.format(mask.sum(), mask.size,
                                          (100.0 * mask.sum())/mask.size)
_test()


# Only 0.4% masked bodes well for the usefulness of this Landsat 8 scene!
# 

# ## Landsat 8 and PS Orthotile Visual Images
# 
# Before we jump into training the classifier, we want to visualize the Landsat 8 and PS Orthotile RGB images to get a feel for what they show.
# 
# To do this, we first create classes that load and store the PS and L8 analytic images, (PSImage and L8Image, respectively). We then create a class that displays an RGB image, taking care of the necessary scaling and masking. We then use these classes to visualize the PS and L8 RGB images.
# 
# ### PS Orthotile Visual Image
# 
# #### Mask from UDM
# 

def get_udm_labels(binary_band):
    return OrderedDict((v, '{0:b}'.format(v).zfill(7))
                       for v in np.unique(binary_band))

def udm_to_mask(udm_array):
    '''Create a mask from the udm.
    
    The description for the information encoded in the bits in the UDM is from:
    https://www.planet.com/docs/spec-sheets/sat-imagery/
    section 7.2
    
    Bit 0: blackfill
    Bit 1: cloud covered
    Bit 2: missing or suspect data in Blue band
    Bit 3: missing or suspect data in Green band
    Bit 4: missing or suspect data in Red band
    Bit 6: missing or suspect data in NIR band

    Pixels with no issues have all bits set to 0, therefore their values are zero.    
    ''' 
    return udm_array != 0

def _test():
    udm = load_band(train_files['udm'])
    mask = udm_to_mask(udm)
    print'{}/{} ({:0.0f}%) masked'.format(mask.sum(), mask.size,
                                          (100.0 * mask.sum())/mask.size)

    udm_dec = decimated(udm, 32)
    udm_labels = get_udm_labels(udm_dec)
    udm_image = ClassifiedImage(band=udm_dec, labels=udm_labels)
    plot_image(udm_image, title='UDM Band', figsize=(5,5))
# _test()


# #### Visual RGB
# 
# Create a few classes that store the analytic and visual images, to simplify processing and visualizing the PS Orthotile image.
# 

PSBands = namedtuple('PSBands', 'b, g, r, nir')

class PSImage(object):    
    def __init__(self, scene_filename, udm_filename, window=None):
        self.scene_filename = scene_filename
        self.udm_filename = udm_filename
        self.window = window

        self._bands = self._load_bands()
        
    def _load_bands(self):
        """Loads a 4-band BGRNir Planet Image file as a list of masked bands.

        The masked bands share the same mask, so editing one band mask will
        edit them all.
        """
        with rasterio.open(self.scene_filename, 'r') as src:
            b, g, r, nir = src.read(window=self.window)
            bands = PSBands(b=b, g=g, r=r, nir=nir)

        with rasterio.open(self.udm_filename, 'r') as src:
            udm = src.read(window=self.window)[0,...]

        mask = udm_to_mask(udm)
        return PSBands(*[np.ma.array(b, mask=mask) for b in bands])

    def rgb_bands(self):
        return [self._bands.r, self._bands.g, self._bands.b]
        
    @property
    def mask(self):
        return self._bands[0].mask
    
    @property
    def bands(self):
        return self._bands


def check_mask(img):
    band_mask = img.mask
    return '{}/{} ({:0.0f}%) masked'.format(band_mask.sum(), band_mask.size,
                                            (100.0 * band_mask.sum())/band_mask.size)

def _test():
    window = ((500,1500),(500,1500))
    print(check_mask(PSImage(train_files['scene'], train_files['udm'], window=window)))

    window = None
    print(check_mask(PSImage(train_files['scene'], train_files['udm'], window=window)))
_test()


# Utility functions: displaying an rgb image

def _linear_scale(ndarray, old_min, old_max, new_min, new_max):
    """Linear scale from old_min to new_min, old_max to new_max.
    
    Values below min/max are allowed in input and output.
    Min/Max values are two data points that are used in the linear scaling.
    """
    #https://en.wikipedia.org/wiki/Normalization_(image_processing)
    return (ndarray - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
# print(linear_scale(np.array([1,2,10,100,256,2560, 2660]), 2, 2560, 0, 256))


class RGBImage(object):
    def __init__(self, bands):
        # bands: list of masked bands in RGB order
        # masked bands should share the same mask
        assert len(bands) == 3
        self.bands = bands

    def imshow(self, ax, alpha=True):
        """Show RGB image with option of convering mask to alpha.
        
        Alters ax in place.

        """
        ax.imshow(self.bands_to_display(alpha=alpha))
        

    def _mask_to_alpha(self):
        band = np.atleast_3d(self.bands[0])
        alpha = np.zeros_like(band)
        alpha[~band.mask] = 1
        return alpha

    def _percentile(self, percentile):
        return np.percentile(np.concatenate([b.compressed() for b in self.bands]),
                             percentile)

    def bands_to_display(self, alpha=False):
        """Converts bands to a normalized, 3-band 3d numpy array for display."""  

        old_min = self._percentile(2)
        old_max = self._percentile(98)
        new_min = 0
        new_max = 1
        scaled = [np.clip(_linear_scale(b.astype(np.float),
                                        old_min, old_max,
                                        new_min, new_max),
                          new_min, new_max)
                  for b in self.bands]

        filled = [b.filled(fill_value=new_min) for b in scaled]

        if alpha:
            filled.append(self._mask_to_alpha())

        return np.dstack(filled)

def _test():
    img = PSImage(train_files['scene'], train_files['udm'], window=None)
    rgb_image = RGBImage([decimated(b) for b in img.rgb_bands()])
    plot_image(rgb_image, title='PS RGB', figsize=(6,6))
# _test()


# ### Landsat 8 Visual Image
# 
# Create a class stores the analytic Landsat 8 image.
# 

L8Bands = namedtuple('L8Bands', 'b2, b3, b4, b5')
        
class L8Image(object):
    def __init__(self, band_filenames, qa_filename, window=None):
        self.band_filenames = band_filenames
        self.qa_filename = qa_filename
        self.window = window

        self._bands = self._load_bands()
    
    def _load_mask(self):
        qa = self._read_band(self.qa_filename)
        return qa_to_mask(qa)
        
    def _load_bands(self):
        def _try_read_band(band_name, mask):
            try:
                filename = self.band_filenames[band_name]
                band_arry = self._read_band(filename)
                band = np.ma.array(band_arry, mask=mask)
            except KeyError:
                # band_name not a key in band_filenames
                band = None
            return band

        mask = self._load_mask()
        return L8Bands(*[_try_read_band(band_name, mask)
                         for band_name in L8Bands._fields])
    
    def _read_band(self, filename):
        with rasterio.open(filename, 'r') as src:
            band = src.read(window=self.window)[0,...]
        return band
    
    def rgb_bands(self):
        rgb_bands = [self._bands.b4, self._bands.b3, self._bands.b2]
        return rgb_bands

    @property
    def mask(self):
        return self._bands[0].mask
    
    @property
    def bands(self):
        return self._bands

def _test():
    img = L8Image(train_l8_files, train_l8_files['qa'], window=None)
    rgb_image = RGBImage([decimated(b) for b in img.rgb_bands()])
    plot_image(rgb_image, title='L8 RGB', figsize=(6,6))
# _test()


# ### Visual Comparison
# 
# Load the PS Orthotile and Landsat 8 images, then compare the visual RGB representation of those images.
# 

def plot_rgb_comparison(ps_files, l8_files, figsize=(15,15)):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   sharex=True, sharey=True,
                                   figsize=figsize)
    for ax in (ax1, ax2):
        ax.set_adjustable('box-forced')

    # PS
    img = PSImage(ps_files['scene'], ps_files['udm'], window=None)
    ps_rgb_image = RGBImage([decimated(b) for b in img.rgb_bands()])
    ps_rgb_image.imshow(ax1)
    ax1.set_title('PS')

    # L8
    rgb_bandnames = ['b4', 'b3', 'b2']
    rgb_files = dict([(k, l8_files[k]) for k in rgb_bandnames])
    l8_img = L8Image(rgb_files, l8_files['qa'], window=None)
    l8_rgb_image = RGBImage([decimated(b) for b in l8_img.rgb_bands()])
    l8_rgb_image.imshow(ax2)
    ax2.set_title('L8')

    plt.tight_layout()

plot_rgb_comparison(train_files, train_l8_files)


# ## Features from Images
# 
# To classify the image, we first create 'feature bands', which are bands that hold each of the classification features, from the L8 and PS analytic images. An important step to creating the feature bands is ensuring that any pixel that is masked in either image is masked in the feature bands.
# 

def build_feature_bands(ps_img, l8_img):
    """Prepare bands representing pixel features and provide feature names.
    
    Takes as input NamedBands
    Returns (1) tuple of bands representing features and (2) feature names
    """  
    # not copying these bands to minimize memory footprints
    features = (ps_img.bands.b, ps_img.bands.g, ps_img.bands.r, ps_img.bands.nir,
                l8_img.bands.b2, l8_img.bands.b3, l8_img.bands.b4, l8_img.bands.b5)
    
    new_mask = ps_img.mask | l8_img.mask
    for band in features:
        band.mask = new_mask

    feature_names = ('PSBlue', 'PSGreen', 'PSRed', 'PSNIR',
                     'L8B2', 'L8B3', 'L8B4', 'L8B5')
    return features, feature_names


def display_feature_bands(bands, names):
    # for this notebook, we know there are 8 features and we will use that
    # knowledge to side-step some logic in organizing subplots
    assert len(bands) == 8 
    
    fig, subplot_axes = plt.subplots(nrows=4, ncols=2,
                                     sharex=True, sharey=True,
                                     figsize=(10,10))
    axes = subplot_axes.flat[:-1]
    delaxis = subplot_axes.flat[-1]
    fig.delaxes(delaxis)
    for ax, band, name in zip(axes, bands, names):
        ax.set_adjustable('box-forced')
        ax.axis('off')

        pcm = ax.imshow(band, alpha=True)
        ax.set_title(name)
        fig.colorbar(pcm, ax=ax,
                     pad=0.05, shrink=0.9)

    plt.tight_layout()

def _test():
#     window = ((500,1500),(500,1500))
    window = None

    ps_img = PSImage(train_files['scene'], train_files['udm'], window=window)
    l8_img = L8Image(train_l8_files, train_l8_files['qa'], window=window)
    feat_bands, feat_names = build_feature_bands(ps_img, l8_img)
    display_feature_bands(feat_bands, feat_names)
_test()


# ## Classify Image
# 
# In this section, we load the labels and train the classifier using the feature bands, then run the prediction on the train feature bands to see how well the features differentiate the classes.
# 
# ### Gold Labels
# 

# this file is created in the 'classify-cart.ipynb' notebook.
# That notebook provides more details on the label dataset and
# must be run first.
    
train_files['gold'] = os.path.join(train_folder, 'CDL_2016_19_prep.tif')
assert os.path.isfile(train_files['gold'])


# copied from classify-cart.ipynb

CLASS_LABELS = {1: 'corn', 5: 'soybeans'}

def load_gold_class_band(gold_filename, class_labels=None, window=None, fill_value=0):
    gold_class_band = _read_window(gold_filename, window)[0,...]
    
    try:
        # mask pixels with a value not in class_labels
        masks = [gold_class_band == val for val in class_labels.keys()]
        mask = np.any(np.dstack(masks), axis=2)
        mask = ~mask
        
        masked_band = np.ma.array(np.ma.array(gold_class_band, mask=mask)                                      .filled(fill_value=fill_value),
                                  mask=mask)
    except AttributeError:
        # mask nothing
        null_mask = np.zeros(gold_class_band.shape, dtype=np.bool)
        masked_band = np.ma.array(gold_class_band, mask=null_mask)

    return masked_band

def _test():
    labels = CLASS_LABELS
    gold_band = load_gold_class_band(train_files['gold'], class_labels=labels)

    gold_dec = decimated(gold_band)
    gold_image = ClassifiedImage(band=gold_dec, labels=labels)
    plot_image(gold_image, title='Gold Band', figsize=(5,5))
# _test()


# ### Train Classifier
# 

# copied from classify-cart.ipynb
def to_X(feature_bands):
    """Convert feature_bands (tuple of bands) to 2d array for working with classifier.
    """
    return np.stack([f.compressed() for f in feature_bands], # exclude masked pixels
                     axis=1)

def to_y(classified_band):
    return classified_band.compressed()

def classified_band_from_y(y, band_mask):
    class_band = np.ma.array(np.zeros(band_mask.shape),
                             mask=band_mask.copy())
    class_band[~class_band.mask] = y
    return class_band


window = None


# Prepare features
ps_img = PSImage(train_files['scene'],
                 train_files['udm'],
                 window=window)
l8_img = L8Image(train_l8_files,
                 train_l8_files['qa'],
                 window=window)
feat_bands, _ = build_feature_bands(ps_img, l8_img)
X = to_X(feat_bands)
print(X.shape)


# Prepare labels
labels = CLASS_LABELS
gold_band = load_gold_class_band(train_files['gold'],
                                 class_labels=labels,
                                 window=window)
gold_band.mask = feat_bands[0].mask #get_mask(feat_bands)
y = to_y(gold_band)
print(y.shape)


# Train classifier on PS + L8 features
clf = DecisionTreeClassifier(random_state=0, max_depth=5)
clf.fit(X, y)


# ### Run Prediction on Train Features
# 

# Run prediction on train features
y_pred = clf.predict(X)


# Display classification results
def plot_predicted_vs_truth(predicted_class_band, gold_class_band,
                            class_labels=None, figsize=(15,15)):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   sharex=True, sharey=True,
                                   figsize=figsize)
    for ax in (ax1, ax2):
        ax.set_adjustable('box-forced')
    
    pred_img = ClassifiedImage(band=predicted_class_band, labels=CLASS_LABELS)
    pred_img.imshow(ax1)
    ax1.set_title('Classifier Predicted Classes')

    gold_img = ClassifiedImage(band=gold_class_band, labels=CLASS_LABELS)
    gold_img.imshow(ax2)
    ax2.set_title('Gold Dataset Classes')
    plt.tight_layout()

pred_band = classified_band_from_y(y_pred, feat_bands[0].mask)
plot_predicted_vs_truth(pred_band, gold_band, class_labels=CLASS_LABELS)


# ### Quantify Performance on Train Features
# 
# Quantify the performance on the train features, which represents the upper limit for classification accuracy when a prediction is run on a new image.
# 

print(classification_report(y,
                            y_pred,
                            target_names=['neither', 'corn', 'soybean']))


# The classifier performance for classifying a pixel as neither, corn, or soybean, calculated on the train features, is an f1-score of 0.84.
# 

# # Compare PS to PS + L8
# 
# How does the accuracy of classification using PS + L8 compare to the accuracy using just PS?

def build_ps_feature_bands(ps_img):
    # not copying these bands to minimize memory footprints
    features = (ps_img.bands.b, ps_img.bands.g, ps_img.bands.r, ps_img.bands.nir)
    feature_names = ('PSBlue', 'PSGreen', 'PSRed', 'PSNIR')
    return features, feature_names

ps_feature_bands = (ps_img.bands.b, ps_img.bands.g, ps_img.bands.r, ps_img.bands.nir)
ps_X = to_X(ps_feature_bands)

# Train classifier
ps_clf = DecisionTreeClassifier(random_state=0, max_depth=5)
ps_clf.fit(ps_X, y)


# Run prediction on train features
ps_y_pred = ps_clf.predict(ps_X)


print(classification_report(y,
                            ps_y_pred,
                            target_names=['neither', 'corn', 'soybean']))





# # Calculate Coverage
# 
# You've defined an AOI, you've specified the image type you are interested and the search query. Great! But what is the coverage of your AOI given your search query? Wouldn't you like to know before you start downloading images?
# 
# This notebook will allow you to answer that question quickly and painlessly.
# 
# Coverage calculation is performed in the UTM [projected coordinate system](http://resources.arcgis.com/en/help/main/10.1/index.html#//003r0000000p000000). The geojson features are defined in the WGS84 [geographic coordinate system](http://resources.arcgis.com/en/help/main/10.1/index.html#//003r00000006000000), which is not a 2D projection. 
# UTM preserves shape and minimizes distortion ([wikipedia](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system))

# Notebook dependencies
from __future__ import print_function

import datetime
import copy
from functools import partial
import os

from IPython.display import display, Image
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from planet import api
from planet.api import filters
import pyproj
import rasterio
from rasterio import features as rfeatures
from shapely import geometry as sgeom
import shapely.ops

get_ipython().magic('matplotlib inline')


# ## Define AOI
# 
# Define the AOI as a geojson polygon. This can be done at [geojson.io](http://geojson.io). If you use geojson.io, only copy the single aoi feature, not the entire feature collection.
# 

aoi = {u'geometry': {u'type': u'Polygon', u'coordinates': [[[-121.3113248348236, 38.28911976564886], [-121.3113248348236, 38.34622533958], [-121.2344205379486, 38.34622533958], [-121.2344205379486, 38.28911976564886], [-121.3113248348236, 38.28911976564886]]]}, u'type': u'Feature', u'properties': {u'style': {u'opacity': 0.5, u'fillOpacity': 0.2, u'noClip': False, u'weight': 4, u'color': u'blue', u'lineCap': None, u'dashArray': None, u'smoothFactor': 1, u'stroke': True, u'fillColor': None, u'clickable': True, u'lineJoin': None, u'fill': True}}}


# this notebook uses rasterio Shapes for processing, so lets convert that geojson to a shape
aoi_shape = sgeom.shape(aoi['geometry'])


# ## Build Request
# 
# Build the Planet API Filter request.
# 
# Customize this code for your own purposes
# 

def build_request(aoi_shape):
    old = datetime.datetime(year=2016,month=6,day=1)
    new = datetime.datetime(year=2016,month=10,day=1)

    query = filters.and_filter(
        filters.geom_filter(sgeom.mapping(aoi_shape)),
        filters.range_filter('cloud_cover', lt=5),
        filters.date_range('acquired', gt=old),
        filters.date_range('acquired', lt=new)
    )
    
    item_types = ['PSOrthoTile']
    return filters.build_search_request(query, item_types)

request = build_request(aoi_shape)
print(request)


# ## Check AOI and Determine Coverage Grid Dimensions
# 
# We convert the AOI to UTM and ensure that it is large enough to include at least a few grid cells 9m x 9m (approximately 3x PS Orthotile resolution). Then we determine the appropriate coverage grid dimensions from the AOI.
# 
# There are a lot of UTM zones, and the UTM zone we project to depends on the location of the AOI. Once this zone is determined, we create a function that can be used to project any shape. We will use that function to project the scene footprints to the same UTM zone once we get them.
# 

# Utility functions: projecting a feature to the appropriate UTM zone

def get_utm_projection_fcn(shape):
    # define projection
    # from shapely [docs](http://toblerity.org/shapely/manual.html#shapely.ops.transform)
    proj_fcn = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'), #wgs84
        _get_utm_projection(shape))
    return proj_fcn


def _get_utm_zone(shape):
    '''geom: geojson geometry'''
    centroid = shape.centroid
    lon = centroid.x
    lat = centroid.y
    
    if lat > 84 or lat < -80:
        raise Exception('UTM Zones only valid within [-80, 84] latitude')
    
    # this is adapted from
    # https://www.e-education.psu.edu/natureofgeoinfo/book/export/html/1696
    zone = int((lon + 180) / 6 + 1)
    
    hemisphere = 'north' if lat > 0 else 'south'
    
    return (zone, hemisphere)


def _get_utm_projection(shape):
    zone, hemisphere = _get_utm_zone(shape)
    proj_str = "+proj=utm +zone={zone}, +{hemi} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(
        zone=zone, hemi=hemisphere)
    return pyproj.Proj(proj_str)


proj_fcn = get_utm_projection_fcn(aoi_shape)
aoi_shape_utm = shapely.ops.transform(proj_fcn, aoi_shape)
print(aoi_shape_utm)


def get_coverage_dimensions(aoi_shape_utm):
    '''Checks that aoi is big enough and calculates the dimensions for coverage grid.'''
    minx, miny, maxx, maxy = aoi_shape_utm.bounds
    width = maxx - minx
    height = maxy - miny
    
    min_cell_size = 9 # in meters, approx 3x ground sampling distance
    min_number_of_cells = 3
    max_number_of_cells = 3000
    
    
    min_dim = min_cell_size * min_number_of_cells
    if height < min_dim:
        raise Exception('AOI height too small, should be {}m.'.format(min_dim))

    if width < min_dim:
        raise Exception('AOI width too small, should be {}m.'.format(min_dim))
    
    def _dim(length):
        return min(int(length/min_cell_size), max_number_of_cells)

    return [_dim(l) for l in (height, width)]


dimensions = get_coverage_dimensions(aoi_shape_utm)
print(dimensions)


# ## Search Planet API
# 
# The client is how we interact with the planet api. It is created with the user-specific api key, which is pulled from $PL_API_KEY environment variable.
# 
# Unless you are expecting over 500 images (in which case, why are you concerned about coverage?), this code doesn't need to be altered.
# 

def get_api_key():
    return os.environ['PL_API_KEY']


# quick check that key is defined
assert get_api_key(), "PL_API_KEY not defined."


def create_client():
    return api.ClientV1(api_key=get_api_key())


def search_pl_api(request, limit=500):
    client = create_client()
    result = client.quick_search(request)
    
    # note that this returns a generator
    return result.items_iter(limit=limit)


# ## Calculate Coverage
# 
# First query the planet api for the items that match the request defined above, then calculate the overlap between each item and the aoi. Finally, convert each overlap to a grid using [`rasterio.rasterize`](https://mapbox.github.io/rasterio/topics/features.html#burning-shapes-into-a-raster), accumulate coverage over the overlap grids, and display the coverage grid.
# 

def get_overlap_shapes_utm(items, aoi_shape):
    '''Determine overlap between item footprint and AOI in UTM.'''
    
    proj_fcn = get_utm_projection_fcn(aoi_shape)
    aoi_shape_utm = shapely.ops.transform(proj_fcn, aoi_shape)

    def _calculate_overlap(item):
        footprint_shape = sgeom.shape(item['geometry'])
        footprint_shape_utm = shapely.ops.transform(proj_fcn, footprint_shape)
        return aoi_shape_utm.intersection(footprint_shape_utm)

    for i in items:
        yield _calculate_overlap(i)


items = search_pl_api(request)

# cache the overlaps as a list so we don't have to refetch items
overlaps = list(get_overlap_shapes_utm(items, aoi_shape))
print(len(overlaps))


# what do overlaps look like?
# lets just look at the first overlap to avoid a long output cell
display(overlaps[0])


def calculate_coverage(overlaps, dimensions, bounds):
    
    # get dimensions of coverage raster
    mminx, mminy, mmaxx, mmaxy = bounds

    y_count, x_count = dimensions
    
    # determine pixel width and height for transform
    width = (mmaxx - mminx) / x_count
    height = (mminy - mmaxy) / y_count # should be negative

    # Affine(a, b, c, d, e, f) where:
    # a = width of a pixel
    # b = row rotation (typically zero)
    # c = x-coordinate of the upper-left corner of the upper-left pixel
    # d = column rotation (typically zero)
    # e = height of a pixel (typically negative)
    # f = y-coordinate of the of the upper-left corner of the upper-left pixel
    # ref: http://www.perrygeo.com/python-affine-transforms.html
    transform = rasterio.Affine(width, 0, mminx, 0, height, mmaxy)
    
    coverage = np.zeros(dimensions, dtype=np.uint16)
    for overlap in overlaps:
        if not overlap.is_empty:
            # rasterize overlap vector, transforming to coverage raster
            # pixels inside overlap have a value of 1, others have a value of 0
            overlap_raster = rfeatures.rasterize(
                    [sgeom.mapping(overlap)],
                    fill=0,
                    default_value=1,
                    out_shape=dimensions,
                    transform=transform)
            
            # add overlap raster to coverage raster
            coverage += overlap_raster
    return coverage


# what is a low-resolution look at the coverage grid?
display(calculate_coverage(overlaps, (6,3), aoi_shape_utm.bounds))


def plot_coverage(coverage):
    fig, ax = plt.subplots()
    cax = ax.imshow(coverage, interpolation='nearest', cmap=cm.viridis)
    ax.set_title('Coverage\n(median: {})'.format(int(np.median(coverage))))
    ax.axis('off')
    
    ticks_min = coverage.min()
    ticks_max = coverage.max()
    cbar = fig.colorbar(cax,ticks=[ticks_min, ticks_max])


plot_coverage(calculate_coverage(overlaps, dimensions, aoi_shape_utm.bounds))


# ## Demo: Comparing Coverage
# 
# We will compare coverage of PS OrthoTiles June and July between 2016 and 2017 for the same aoi.
# 

demo_aoi = aoi  # use the same aoi that was used before

demo_aoi_shape = sgeom.shape(demo_aoi['geometry'])

proj_fcn = get_utm_projection_fcn(demo_aoi_shape)
demo_aoi_shape_utm = shapely.ops.transform(proj_fcn, demo_aoi_shape)
demo_dimensions = get_coverage_dimensions(demo_aoi_shape_utm)                               


# Parameterize our search request by start/stop dates for this comparison
def build_request_by_dates(aoi_shape, old, new):
    query = filters.and_filter(
        filters.geom_filter(sgeom.mapping(aoi_shape)),
        filters.range_filter('cloud_cover', lt=5),
        filters.date_range('acquired', gt=old),
        filters.date_range('acquired', lt=new)
    )
    
    item_types = ['PSOrthoTile']
    return filters.build_search_request(query, item_types)  


request_2016 = build_request_by_dates(demo_aoi_shape,
                                      datetime.datetime(year=2016,month=6,day=1),
                                      datetime.datetime(year=2016,month=8,day=1))                                    
items = search_pl_api(request_2016)
overlaps = list(get_overlap_shapes_utm(items, demo_aoi_shape))
plot_coverage(calculate_coverage(overlaps, demo_dimensions, demo_aoi_shape_utm.bounds))


request_2017 = build_request_by_dates(demo_aoi_shape,
                                      datetime.datetime(year=2017,month=6,day=1),
                                      datetime.datetime(year=2017,month=8,day=1))
items = search_pl_api(request_2017)
overlaps = list(get_overlap_shapes_utm(items, demo_aoi_shape))
plot_coverage(calculate_coverage(overlaps, demo_dimensions, demo_aoi_shape_utm.bounds))


# Median coverage over 2 months has increased from 2 to 69! That's a decrease in average revisit rate from 1/month to over 1/day. That's what a constellation of over 100 satellites will do for you!
# 




# ## Detect ships in Planet data
# This notebook demonstrates how to detect and count objects in satellite imagery using algorithms from Python's scikit-image library. In this example, we'll look for ships in a small area in the San Francisco Bay and generate a PNG of each ship with an outline around it.
# 

# ### Input Parameters
# This is a sample image that was generated using the Clip and Ship API. To test this with your own image, replace the parameters below.
# 

sample_data_file_name = 'data/1056417_2017-03-08_RE3_3A_Visual_clip.tif'


# ### Original image
# Below is a predefined image that has been clipped from the Planet API using [Clip and Ship](https://www.planet.com/docs/reference/clips-api/). This is the image that we are going to detect ships in.
# 

import skimage.io
from IPython.display import Image

# Read image into scimage package
img = skimage.io.imread(sample_data_file_name)
skimage.io.imsave('output/original.png', img)

# Display original image
display(Image(filename='output/original.png'))


# ### Run the ship detection algorithm 
# 

import json
from osgeo import gdal, osr
import numpy
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops

# Prepare result structure
result = {
    "ship_count": 0,
    "ships": []
}

# Open image with gdal
ds = gdal.Open(sample_data_file_name)
xoff, a, b, yoff, d, e = ds.GetGeoTransform()

# Get projection information from source image
ds_proj = ds.GetProjectionRef()
ds_srs = osr.SpatialReference(ds_proj)

# Get the source image's geographic coordinate system (the 'GEOGCS' node of ds_srs)
geogcs = ds_srs.CloneGeogCS()

# Set up a transformation between projected coordinates (x, y) & geographic coordinates (lat, lon)
transform = osr.CoordinateTransformation(ds_srs, geogcs)

# Convert multi-channel image it into red, green and blueb[, alpha] channels 
red, green, blue, alpha = numpy.rollaxis(numpy.array(img), axis=-1)

# Mask: threshold + stops canny detecting image boundary edges
mask = red > 75

# Create mask for edge detection
skimage.io.imsave('output/mask.png', mask * 255)

# Use Felzenszwalb algo to find segements
segments_fz = felzenszwalb(numpy.dstack((mask, mask, mask)),
                               scale=5000,
                               sigma=3.1,
                               min_size=25) 

# Build labeled mask to show where ships were dectected
segmented_img = mark_boundaries(mask, segments_fz)
skimage.io.imsave('output/mask_labeled.png', segmented_img)

# Count ships and save image of each boat clipped from masked image
for idx, ship in enumerate(regionprops(segments_fz)):
    
    # If area matches that of a stanard ship, count it
    if (ship.area >= 300 and ship.area <= 10000):
        
        # Incrment count
        result['ship_count'] += 1
        
        # Create ship thumbnail
        x, y = (int(numpy.average([ship.bbox[0],
                                ship.bbox[2]])),
                                int(numpy.average([ship.bbox[1],
                                ship.bbox[3]])))
        sx, ex = max(x - 35, 0), min(x + 35, img.shape[0] - 1)
        sy, ey = max(y - 35, 0), min(y + 35, img.shape[1] - 1)
        img_ship = img[sx:ex, sy:ey]
        skimage.io.imsave('output/ship-%s.png' % str(idx + 1), img_ship)

        # Get global coordinates from pixel x, y coords
        projected_x = a * y + b * x + xoff
        projected_y = d * y + e * x + yoff
        
        # Transform from projected x, y to geographic lat, lng
        (lat, lng, elev) = transform.TransformPoint(projected_x, projected_y)
        
        # Add ship to results cluster
        result["ships"].append({
            "id": idx + 1,
            "lat": lat,
            "lng": lng
        })

# Display results
print(json.dumps(result, indent=2))

#Display mask used for ship detection.
display(Image(filename='output/mask.png'))

# Display labled mask where we detected ships
display(Image(filename='output/mask_labeled.png'))

# Display each individual ship cropped out of the original image
for idx,ship in enumerate(result['ships']):
    print("Ship "+ str(idx + 1)) 
    display(Image(filename='output/ship-' + str(idx + 1) + '.png'))


