import os
from datetime import datetime, timedelta

import numpy as np
import psycopg2
from dotenv import load_dotenv
from sentinelhub import MimeType, CRS, SentinelHubRequest, SentinelHubDownloadClient, DataCollection, bbox_to_dimensions
from sentinelhub import UtmZoneSplitter, SHConfig
from sentinelhub import read_data
from shapely.geometry import shape
from tensorflow.keras.initializers import RandomNormal, HeUniform
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, Dropout, BatchNormalization, LeakyReLU, \
    Concatenate, ReLU
from tensorflow.keras.models import Model

load_dotenv()

resolution = 40
needed_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B08', 'B09', "B10", 'B11', 'B12']
parallel = 20

geo_data_dir = "geo_data"
district_geojson_dir = os.path.join(geo_data_dir, "district")
model_file = os.path.join("model_saves", "unet_epochs_50.h5")

config = SHConfig()

config.instance_id = os.environ.get('INSTANCE_ID')
config.sh_client_id = os.environ.get('CLIENT_ID')
config.sh_client_secret = os.environ.get('CLIENT_SECRET')
config.save()

evalscript_all_bands = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
                units: "DN"
            }],
            output: {
                bands: 13,
                sampleType: "INT16",
                mosaicking: "SIMPLE",
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B01,
                sample.B02,
                sample.B03,
                sample.B04,
                sample.B05,
                sample.B06,
                sample.B07,
                sample.B08,
                sample.B8A,
                sample.B09,
                sample.B10,
                sample.B11,
                sample.B12];
    }
"""

label_enum = {
    0: "Water",
    1: "Artificial Bare Ground",
    2: "Artificial Natural Ground",
    3: "Water",
    4: "Woody",
    5: "Non Woody Cultivated",
    6: "Non Woody Natural",
}


def get_sattelite_bands_request(time_interval, bbox, size):
    return SentinelHubRequest(
        evalscript=evalscript_all_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=time_interval,
                mosaicking_order='leastCC'
            )],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=size,
        config=config
    )


def get_satellite_images(bbox_list, bbox_info, day_slots):
    tiles = []
    for i in range(len(bbox_list)):
        bbox = bbox_list[i]
        info = info_list[i]
        bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
        list_of_requests = [get_sattelite_bands_request(slot, bbox, bbox_size) for slot in day_slots]
        list_of_requests = [request.download_list[0] for request in list_of_requests]

        data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5)
        band_stack = []
        for band_ind in range(len(needed_bands)):
            image_stack = []
            for image in data:
                image = np.clip(image * 2.5 / 255, 0, 1)
                image_stack.append(image[:, :, band_ind])
            band_stack.append(image_stack)

        band_stack = np.stack(band_stack)
        tiles.append(band_stack)

    return np.stack(tiles)


def get_sort_band(img_arr, axs=2):
    '''
    numpy  sorting
    '''

    sorted_img = np.sort(img_arr, axis=axs)
    return sorted_img


def generate_cld_less(sorted_arr, min_limit=0, max_limit=-2):
    avg_arr = np.mean(sorted_arr[:, :, min_limit:max_limit, :, :], axis=2)
    # print(avg_arr.shape)
    return avg_arr


def remove_clds(image_stack):
    image_stack = get_sort_band(image_stack)
    image_stack = generate_cld_less(image_stack)
    return image_stack


def get_date_slots(no_days_back=90, n_chunks=3):
    start = datetime.now() - timedelta(days=no_days_back)
    end = datetime.now()
    tdelta = (end - start) / n_chunks
    edges = [(start + i * tdelta).date().isoformat() for i in range(n_chunks)]
    slots = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    slots.append((edges[-1], datetime.now().date().isoformat()))
    return slots


def read_json_and_break_into_bbox(geo_json_file, distance_of_image=(2560, 2560)):
    geo_json = read_data(geo_json_file)
    aoi = shape(geo_json["features"][0]["geometry"])

    utm_zone_splitter = UtmZoneSplitter([aoi], CRS.WGS84, distance_of_image)
    return utm_zone_splitter


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    init = HeUniform()
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g


def define_decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.4)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = ReLU()(g)
    return g


def define_generator(latent_size, image_shape=(128, 128, 2)):
    init = RandomNormal(stddev=0.02)
    input_image = Input(shape=image_shape)
    #     style_image = Input(shape=image_shape)
    # stack content and style images
    #     stacked_layer = Concatenate()([content_image, style_image])
    # encoder model
    e1 = define_encoder_block(input_image, 32, batchnorm=False)
    e2 = define_encoder_block(e1, 64)
    e3 = define_encoder_block(e2, 128)
    e4 = define_encoder_block(e3, 256)
    e5 = define_encoder_block(e4, 256)
    e6 = define_encoder_block(e5, 256)
    # e7 = define_encoder_block(e6, 512)
    # bottleneck layer
    b = Conv2D(latent_size, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e6)
    b = ReLU()(b)
    # decoder model
    # d1 = define_decoder_block(b, e7, 512)
    d2 = define_decoder_block(b, e6, 256)
    d3 = define_decoder_block(d2, e5, 256)
    d4 = define_decoder_block(d3, e4, 256, dropout=False)
    d5 = define_decoder_block(d4, e3, 128, dropout=False)
    d6 = define_decoder_block(d5, e2, 64, dropout=False)
    d7 = define_decoder_block(d6, e1, 32, dropout=False)
    # output layer
    g = Conv2DTranspose(7, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    # o = Reshape((256,256))(g)
    h = Activation('relu')(g)
    out_image = h
    model = Model(inputs=input_image, outputs=out_image, name='generator')
    return model


def transform_to_model_input(image_stack):
    return np.moveaxis(image_stack[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12], :, :], 1, -1)


def calculate_ndvi(image_stack):
    ndvi = (image_stack[:, 7, :, :] - image_stack[:, 3, :, :]) / (image_stack[:, 7, :, :] + image_stack[:, 3, :, :])
    return np.mean(ndvi)


def calculate_burn_index(image_stack):
    burn_index = (image_stack[:, 8, :, :] - image_stack[:, 12, :, :]) / (
            image_stack[:, 8, :, :] + image_stack[:, 12, :, :])
    return np.mean(burn_index)


def prediction_to_area(model_pred, factor=1e-4):
    model_pred = np.argmax(model_pred, axis=3)
    return [
        np.sum(model_pred == 0) * factor + np.sum(model_pred == 3) * factor,
        np.sum(model_pred == 1) * factor,
        np.sum(model_pred == 2) * factor,
        np.sum(model_pred == 4) * factor,
        np.sum(model_pred == 5) * factor,
        np.sum(model_pred == 6) * factor,
    ]


def insert_stat_to_database(stats, district):
    """ insert a new vendor into the vendors table """
    sql = """INSERT INTO forest_stats_foreststats(district, created_at, updated_at, water, artificial_bare_ground, artificial_natural_ground, 
    woody, non_woody_cultivated, non_woody_natural, mean_ndvi, mean_burn_index)
             VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id;"""
    conn = None
    stat_id = None
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, (
            district, datetime.now(), datetime.now(), stats["Water"], stats["Artificial_Bare_Ground"],
            stats["Artificial_Natural_Ground"],
            stats["Woody"],
            stats["Non_Woody_Cultivated"], stats["Non_Woody_Natural"], stats["Mean_NDVI"], stats["Mean_burn_index"],))
        # get the generated id back
        stat_id = cur.fetchone()[0]
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error.message)
    finally:
        if conn is not None:
            conn.close()

    return stat_id


if __name__ == '__main__':
    district_geojsons = os.listdir(district_geojson_dir)
    model = define_generator(32, (256, 256, 12))
    model.load_weights(model_file)

    day_slots = get_date_slots()

    for district in district_geojsons:
        district_name = district.split(".")[0]
        geo_json_file = os.path.join(district_geojson_dir, district)
        bbox_splitter = read_json_and_break_into_bbox(geo_json_file, distance_of_image=(2560 * 4, 2560 * 4))

        bbox_list = bbox_splitter.get_bbox_list()
        info_list = bbox_splitter.get_info_list()

        area_stat_lis = []
        ndvi_lis = []
        burn_index_lis = []
        for i in range(len(bbox_list) // parallel):
            bbox_set = bbox_list[i * parallel: (i + 1) * parallel]
            info_set = info_list[i * parallel: (i + 1) * parallel]

            if i == len(bbox_list) // parallel - 1:
                bbox_set = bbox_list[i * parallel:]
                info_set = info_list[i * parallel:]

            satellite_image_stack_district = get_satellite_images(bbox_set, info_set, day_slots)
            cld_less_images = remove_clds(satellite_image_stack_district)
            land_cover_predictions = model.predict(transform_to_model_input(cld_less_images))
            area_stats = prediction_to_area(land_cover_predictions)
            area_stat_lis.append(area_stats)
            ndvi_lis.append(calculate_ndvi(cld_less_images))
            burn_index_lis.append(calculate_burn_index(cld_less_images))

        area_stat_lis = np.sum(area_stat_lis, axis=0)
        stat_lis = {
            "Water": area_stat_lis[0],
            "Artificial_Bare_Ground": area_stat_lis[1],
            "Artificial_Natural_Ground": area_stat_lis[2],
            "Woody": area_stat_lis[3],
            "Non_Woody_Cultivated": area_stat_lis[4],
            "Non_Woody_Natural": area_stat_lis[5],
            "Mean_NDVI": np.mean(ndvi_lis),
            "Mean_burn_index": np.mean(burn_index_lis),
        }

        print(stat_lis)
        insert_stat_to_database(stat_lis, district_name)
