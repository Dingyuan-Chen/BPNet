import numpy as np
import time
import json
import os.path
from tqdm import tqdm
import functools

import rasterio
from osgeo import gdal, ogr
from osgeo import osr
import overpy
from pyproj import Proj, transform, Transformer
import fiona
import fiona.crs
import shapely.geometry
import shapely.ops

from . import polygon_utils
from . import math_utils
from . import print_utils

# --- Params --- #

QUERY_BASE = \
    """
    <osm-script timeout="900" element-limit="1073741824">
      <union>
        <query type="way">
          <has-kv k="{0}"/>
          <bbox-query s="{1}" w="{2}" n="{3}" e="{4}"/>
        </query>
        <recurse type="way-node" into="nodes"/>
      </union>
      <print/>
    </osm-script>
    """

WGS84_WKT = """
    GEOGCS["GCS_WGS_1984",
        DATUM["WGS_1984",
            SPHEROID["WGS_84",6378137,298.257223563]],
                PRIMEM["Greenwich",0],
        UNIT["Degree",0.017453292519943295]]
        """

CRS = {'no_defs': True, 'ellps': 'WGS84', 'datum': 'WGS84', 'proj': 'longlat'}


# --- --- #


def get_coor_in_space(image_filepath):
    """

    :param image_filepath: Path to geo-referenced tif image
    :return: coor in original space and in wsg84 spatial reference and original geotransform
    :return: geo transform (x_min, res, 0, y_max, 0, -res)
    :return: [[OR_x_min,OR_y_min,OR_x_max,OR_y_max],[TR_x_min,TR_y_min,TR_x_max,TR_y_max]]
    """
    # print(" get_coor_in_space(image_filepath)")
    ds = gdal.Open(image_filepath)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()

    x_min = gt[0]
    y_min = gt[3] + width * gt[4] + height * gt[5]
    x_max = gt[0] + width * gt[1] + height * gt[2]
    y_max = gt[3]

    prj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=prj)

    coor_sys = srs.GetAttrValue("PROJCS|AUTHORITY", 1)

    if coor_sys is None:
        coor_sys = srs.GetAttrValue("GEOGCS|AUTHORITY", 1)

    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(WGS84_WKT)

    # print(srs, new_cs)
    transform = osr.CoordinateTransformation(srs, new_cs)

    lat_long_min = transform.TransformPoint(x_min, y_min)
    lat_long_max = transform.TransformPoint(x_max, y_max)

    coor = [[x_min, y_min, x_max, y_max], [lat_long_min[0], lat_long_min[1], lat_long_max[0], lat_long_max[1]]]
    return coor, gt, coor_sys


def get_osm_data(coor_query):
    """

    :param coor_query: [x_min, min_z, x_max, y_max]
    :return: OSM query result
    """
    api = overpy.Overpass()
    query_buildings = QUERY_BASE.format("building", coor_query[1], coor_query[0], coor_query[3], coor_query[2])
    query_successful = False
    wait_duration = 60
    result = None
    while not query_successful:
        try:
            result = api.query(query_buildings)
            query_successful = True
        except overpy.exception.OverpassGatewayTimeout or overpy.exception.OverpassTooManyRequests or ConnectionResetError:
            print("OSM server overload. Waiting for {} seconds before querying again...".format(wait_duration))
            time.sleep(wait_duration)
            wait_duration *= 2  # Multiply wait time by 2 for the next time
    return result


def proj_to_epsg_space(nodes, coor_sys):
    original = Proj(CRS)
    destination = Proj(init='EPSG:{}'.format(coor_sys))
    polygon = []
    for node in nodes:
        polygon.append(transform(original, destination, node.lon, node.lat))
    return np.array(polygon)


def compute_epsg_to_image_mat(coor, gt):
    x_min = coor[0][0]
    y_max = coor[0][3]

    transform_mat = np.array([
        [gt[1], 0, 0],
        [0, gt[5], 0],
        [x_min, y_max, 1],
    ])
    return np.linalg.inv(transform_mat)


def compute_image_to_epsg_mat(coor, gt):
    x_min = coor[0][0]
    y_max = coor[0][3]

    transform_mat = np.array([
        [gt[1], 0, 0],
        [0, gt[5], 0],
        [x_min, y_max, 1],
    ])
    return transform_mat


def apply_transform_mat(polygon_epsg_space, transform_mat):
    polygon_epsg_space_homogeneous = math_utils.to_homogeneous(polygon_epsg_space)
    polygon_image_space_homogeneous = np.matmul(polygon_epsg_space_homogeneous, transform_mat)
    polygon_image_space = math_utils.to_euclidian(polygon_image_space_homogeneous)
    return polygon_image_space


def get_polygons_from_osm(image_filepath, tag="", ij_coords=True):
    coor, gt, coor_system = get_coor_in_space(image_filepath)
    transform_mat = compute_epsg_to_image_mat(coor, gt)
    osm_data = get_osm_data(coor[1])

    polygons = []
    for way in osm_data.ways:
        if way.tags.get(tag, "n/a") != 'n/a':
            polygon = way.nodes
            polygon_epsg_space = proj_to_epsg_space(polygon, coor_system)
            polygon_image_space = apply_transform_mat(polygon_epsg_space, transform_mat)
            if ij_coords:
                polygon_image_space = polygon_utils.swap_coords(polygon_image_space)
            polygons.append(polygon_image_space)

    return polygons


def get_polygons_from_shapefile(image_filepath, input_shapefile_filepath, progressbar=True):
    def process_one_polygon(polygon):
        assert len(polygon.shape) == 2, "polygon should have shape (n, d), not {}".format(polygon.shape)
        if 2 < polygon.shape[1]:
            print_utils.print_warning(
                "WARNING: polygon from shapefile has shape {}. Will discard extra values to have polygon with shape ({}, 2)".format(
                    polygon.shape, polygon.shape[0]))
            polygon = polygon[:, :2]
        polygon_epsg_space = polygon
        polygon_image_space = apply_transform_mat(polygon_epsg_space, transform_mat)
        polygon_image_space = polygon_utils.swap_coords(polygon_image_space)
        polygons.append(polygon_image_space)

        # Extract properties:
        if "properties" in parsed_json:
            properties = parsed_json["properties"]
            properties_list.append(properties)

    coor, gt, coor_system = get_coor_in_space(image_filepath)
    transform_mat = compute_epsg_to_image_mat(coor, gt)

    file = ogr.Open(input_shapefile_filepath)
    assert file is not None, "File {} does not exist!".format(input_shapefile_filepath)
    shape = file.GetLayer(0)
    feature_count = shape.GetFeatureCount()
    polygons = []
    properties_list = []
    if progressbar:
        iterator = tqdm(range(feature_count), desc="Reading features", leave=False)
    else:
        iterator = range(feature_count)
    for feature_index in iterator:
        feature = shape.GetFeature(feature_index)
        raw_json = feature.ExportToJson()
        parsed_json = json.loads(raw_json)

        # Extract polygon:
        geometry = parsed_json["geometry"]
        if geometry["type"] == "Polygon":
            polygon = np.array(geometry["coordinates"][0])  # TODO: handle polygons with holes (remove [0])
            process_one_polygon(polygon)
        if geometry["type"] == "MultiPolygon":
            for individual_coordinates in geometry["coordinates"]:
                process_one_polygon(np.array(individual_coordinates[0]))  # TODO: handle polygons with holes (remove [0])

    if properties_list:
        return polygons, properties_list
    else:
        return polygons


def create_ogr_polygon(polygon, transform_mat):
    polygon_swapped_coords = polygon_utils.swap_coords(polygon)
    polygon_epsg = apply_transform_mat(polygon_swapped_coords, transform_mat)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in polygon_epsg:
        ring.AddPoint(coord[0], coord[1])

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly.ExportToWkt()


def create_ogr_polygons(polygons, transform_mat):
    ogr_polygons = []
    for polygon in polygons:
        ogr_polygons.append(create_ogr_polygon(polygon, transform_mat))
    return ogr_polygons


def save_image_as_geotiff(save_filepath, image, source_geotiff_filepath):
    # Get geo info from source image:
    source_ds = gdal.Open(source_geotiff_filepath)
    if source_ds is None:
        raise FileNotFoundError(f"Could not load source file {source_geotiff_filepath}")
    source_gt = source_ds.GetGeoTransform()
    source_prj = source_ds.GetProjection()

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(save_filepath, image.shape[1], image.shape[0], image.shape[2])
    outdata.SetGeoTransform(source_gt)  ##sets same geotransform as input
    outdata.SetProjection(source_prj)  ##sets same projection as input
    for i in range(image.shape[2]):
        outdata.GetRasterBand(i + 1).WriteArray(image[..., i])
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    band = None
    ds = None


def save_shapefile_from_polygons(polygons, image_filepath, output_shapefile_filepath, properties_list=None):
    """
    https://gis.stackexchange.com/a/52708/8104
    """
    assert type(polygons) == list and type(polygons[0]) == np.ndarray and \
           len(polygons[0].shape) == 2 and polygons[0].shape[1] == 2, \
        "polygons should be a list of numpy arrays with shape (N, 2)"
    if properties_list is not None:
        assert len(polygons) == len(properties_list), "polygons and properties_list should have the same length"

    coor, gt, coor_system = get_coor_in_space(image_filepath)
    transform_mat = compute_image_to_epsg_mat(coor, gt)
    # Convert polygons to ogr_polygons
    ogr_polygons = create_ogr_polygons(polygons, transform_mat)

    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(output_shapefile_filepath)

    # create the spatial reference, WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    field_name_list = []
    field_type_list = []
    if properties_list is not None:
        for properties in properties_list:
            for (key, value) in properties.items():
                if key not in field_name_list:
                    field_name_list.append(key)
                    field_type_list.append(type(value))
    for (name, py_type) in zip(field_name_list, field_type_list):
        if py_type == int:
            ogr_type = ogr.OFTInteger
        elif py_type == float:
            print("is float")
            ogr_type = ogr.OFTReal
        elif py_type == str:
            ogr_type = ogr.OFTString
        else:
            ogr_type = ogr.OFTInteger
        layer.CreateField(ogr.FieldDefn(name, ogr_type))

    defn = layer.GetLayerDefn()

    for index in range(len(ogr_polygons)):
        ogr_polygon = ogr_polygons[index]
        if properties_list is not None:
            properties = properties_list[index]
        else:
            properties = {}

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        for (key, value) in properties.items():
            feat.SetField(key, value)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkt(ogr_polygon)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


def save_shapefile_from_shapely_polygons(polygons, image_filepath, output_shapefile_filepath):
    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }
    shp_crs = "EPSG:4326"
    shp_srs = Proj(shp_crs)
    raster = rasterio.open(image_filepath)
    raster_srs = Proj(raster.crs)
    raster_proj = lambda x, y: raster.transform * (x, y)
    # shp_proj = functools.partial(transform, raster_srs, shp_srs)
    shp_proj = Transformer.from_proj(raster_srs, shp_srs).transform

    # Write a new Shapefile
    os.makedirs(os.path.dirname(output_shapefile_filepath), exist_ok=True)
    with fiona.open(output_shapefile_filepath, 'w', driver='ESRI Shapefile', schema=schema, crs=fiona.crs.from_epsg(4326)) as c:
        for id, polygon in enumerate(polygons):
            # print("---")
            # print(polygon)
            raster_polygon = shapely.ops.transform(raster_proj, polygon)
            # print(raster_polygon)
            # shp_polygon = shapely.ops.transform(shp_proj, raster_polygon)
            # print(shp_polygon)

            wkt_polygon = shapely.geometry.mapping(raster_polygon)

            c.write({
                'geometry': wkt_polygon,
                'properties': {'id': id},
            })


def indices_of_biggest_intersecting_polygon(polygon_list):
    """
    Assumes polygons which intersect follow each other on the order given by polygon_list.
    This avoids the huge complexity of looking for an intersection between every polygon.

    :param ori_gt_polygons:
    :return:
    """
    keep_index_list = []

    current_cluster = []  # Indices of the polygons belonging to the current cluster (their union has one component)

    for index, polygon in enumerate(polygon_list):
        #  First, check if polygon intersects with current_cluster:
        current_cluster_polygons = [polygon_list[index] for index in current_cluster]
        is_intersection = polygon_utils.check_intersection_with_polygons(polygon, current_cluster_polygons)
        if is_intersection:
            # Just add polygon to the cluster, nothing else to do
            current_cluster.append(index)
        else:
            # This mean the current polygon is part of the next cluster.
            # First, find the biggest polygon in the current cluster
            cluster_max_index = 0
            cluster_max_area = 0
            for cluster_polygon_index in current_cluster:
                cluster_polygon = polygon_list[cluster_polygon_index]
                area = polygon_utils.polygon_area(cluster_polygon)
                if cluster_max_area < area:
                    cluster_max_area = area
                    cluster_max_index = cluster_polygon_index
            # Add index of the biggest polygon to the keep_index_list:
            keep_index_list.append(cluster_max_index)

            # Second, create a new cluster with the current polygon index
            current_cluster = [index]

    return keep_index_list


def get_pixelsize(filepath):
    raster = gdal.Open(filepath)
    gt = raster.GetGeoTransform()
    pixelsize_x = gt[1]
    pixelsize_y = -gt[5]
    pixelsize = (pixelsize_x + pixelsize_y) / 2
    return pixelsize


def crop_shapefile(input_filepath, mask_filepath, output_filepath):
    shp_mask_filepath = os.path.join(os.path.dirname(input_filepath), "mask.shp")
    # ogr2ogr.main(["", "-f", "ESRI Shapefile", shp_mask_filepath, mask_filepath])
    # # ogr2ogr.main(["", "-f", "KML", "-clipsrc", mask_filepath, output_filepath, input_filepath])
    # # script_filepath = os.path.join(os.path.dirname(__file__), "crop_shp_with_shp.sh")
    # # subprocess.Popen(["ogr2ogr", "-clipsrc", mask_filepath, output_filepath, input_filepath])
    #
    # print(input_filepath)
    # print(mask_filepath)
    # print(output_filepath)
    # callstr = ['ogr2ogr',
    #            "-overwrite",
    #            "-t_srs",
    #            "EPSG:27700",
    #            '-clipsrc',
    #            shp_mask_filepath,
    #            output_filepath,
    #            input_filepath,
    #            "-skipfailures"]
    # proc = subprocess.Popen(callstr, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stdout, stderr = proc.communicate()
    # print(stdout)
    # print(stderr)

    input_file = ogr.Open(input_filepath)
    assert input_file is not None, "File {} does not exist!".format(input_filepath)
    input_layer = input_file.GetLayer(0)
    # for i in range(input_layer.GetFeatureCount()):
    #     feature = input_layer.GetFeature(i)
    #     raw_json = feature.ExportToJson()
    #     parsed_json = json.loads(raw_json)
    #     print(parsed_json)
    #     break

    mask_file = ogr.Open(shp_mask_filepath)
    assert mask_file is not None, "File {} does not exist!".format(shp_mask_filepath)
    mask_layer = mask_file.GetLayer(0)
    print(mask_layer.GetFeatureCount())
    feature = mask_layer.GetFeature(0)
    raw_json = feature.ExportToJson()
    parsed_json = json.loads(raw_json)
    print(parsed_json)

    # create empty result layer
    ogrGeometryType = ogr.Geometry(ogr.wkbPolygon)
    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    outDs = outDriver.CreateDataSource(output_filepath)
    outLayer = outDs.CreateLayer('', None, ogr.wkbPolygon)

    input_layer.Intersection(mask_layer, outLayer, options=["SKIP_FAILURES=YES"])


def main():
    main_dirpath = "/workspace/data/stereo_dataset/raw/leibnitz"
    image_filepath = os.path.join(main_dirpath, "leibnitz_ortho_ref_RGB.tif")
    input_shapefile_filepath = os.path.join(main_dirpath, "Leibnitz_buildings_ref.shp")
    output_shapefile_filepath = os.path.join(main_dirpath, "Leibnitz_buildings_ref.shifted.shp")

    polygons, properties_list = get_polygons_from_shapefile(image_filepath, input_shapefile_filepath)
    print(polygons[0])
    print(properties_list[0])

    # Add shift
    shift = np.array([0, 0])
    shifted_polygons = [polygon + shift for polygon in polygons]
    print(shifted_polygons[0])

    # Save shapefile
    save_shapefile_from_polygons(shifted_polygons, image_filepath, output_shapefile_filepath, properties_list=properties_list)


if __name__ == "__main__":
    main()
