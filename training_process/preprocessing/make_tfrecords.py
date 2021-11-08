"""
This script takes the folder in ./data containing annotations csv files and images and produces 
TFRecords out of them, which are stored inside ./tfrecords
"""

import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm

"""
Define TFRecords helper functions
"""
def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class_text_to_label = {
    "Adidas":1,
    "AppleInc":2,
    "Chanel":3,
    "CocaCola":4,
    "Emirates":5,
    "HardRockCafe":6,
    "MercedesBenz":7,
    "NFL":8,
    "Nike":9,
    "Pepsi":10,
    "Puma":11,
    "Starbucks":12,
    "TheNorthFace":13,
    "Toyota":14,
    "UnderArmour":15
}

def create_example(image, path, example):
    feature = {
        "image/encoded": image_feature(image),
        "image/filename": bytes_feature(example["filename"]),
        "image/format": bytes_feature("jpeg"),
        "image/height": int64_feature(example["height"]),
        "image/object/bbox/xmax": float_feature(example["xmax"]),
        "image/object/bbox/xmin": float_feature(example["xmin"]),
        "image/object/bbox/ymax": float_feature(example["ymax"]),
        "image/object/bbox/ymin": float_feature(example["ymin"]),
        "image/object/class/label": int64_feature(class_text_to_label[example["class"]]),
        "image/object/class/text": bytes_feature(example["class"]),
        "image/width": int64_feature(example["width"])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_tfrecord_fn(example):
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/object/class/text": tf.io.FixedLenFeature([], tf.string),
        "image/object/class/label": tf.io.FixedLenFeature([], tf.int64),
        "image/object/bbox/xmin": tf.io.FixedLenFeature([], tf.float32),
        "image/object/bbox/ymin": tf.io.FixedLenFeature([], tf.float32),
        "image/object/bbox/xmax": tf.io.FixedLenFeature([], tf.float32),
        "image/object/bbox/ymax": tf.io.FixedLenFeature([], tf.float32)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example



"""
Define function to generate data in TFRecord format
"""

def produce_tfrecords(images_dir):
    """
    Given the directory in which the annotations csv file and the images are located, produce a
    tfrecord file in the tfrecords folder
    """
    annotations_path = images_dir + "/_annotations_clean.csv"
    annotations = pd.read_csv(annotations_path)

    print(f"Writing TFRecord file for {images_dir}")
    
    tfrecords_dir = f"./tfrecords/{images_dir[7:]}"
    try:
        os.makedirs(tfrecords_dir, exist_ok=True)
    except:
        pass

    tfrecord_path = tfrecords_dir + "/logos.tfrecord"
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for i in tqdm(range(len(annotations))):
            sample = annotations.iloc[i]
            image_filename = sample["filename"]
            image_path = f"{images_dir}/{image_filename}"
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
            example = create_example(image, image_path, sample)
            writer.write(example.SerializeToString())

"""
Produce TFRecords for each folder in ./data
"""
splits = ["train", "valid", "test"]
for folder in os.listdir("./data"):
    if folder[-4:] == ".zip" or not "Batch" in folder:
        continue
    for split in splits:
        images_dir = f"./data/{folder}/{split}"
        produce_tfrecords(images_dir)
    print(f"Finished producing tfrecords for ./data/{folder} \n")
print("Finished producing tfrecords for all folders! \n")
