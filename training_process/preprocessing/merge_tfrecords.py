"""
This script takes tfrecords from ./tfrecords and merge them into a single folder for each train, valid 
and test splits inside ./output_tfrecords
"""

import os
import shutil
import tensorflow as tf
import numpy as np

splits = ["train", "valid", "test"]

"""
Create train, valid and test inside ./output_tfrecords and move .pbtxt file inside each split folder
"""
source_file = "./logos_label_map.pbtxt"
for split in splits:
    dest_path = "./output_tfrecords/" + split
    try:
        os.makedirs(dest_path, exist_ok=True)
    except:
        pass
    shutil.copy(source_file, dest_path)


"""
Create list of tfrecord files for each split
"""
tfrecords_files = {"train":[],
                   "valid":[],
                   "test":[]}

for folder in os.listdir("./tfrecords"):
    for split in splits:
        tfrecord_path = f"./tfrecords/{folder}/{split}/logos.tfrecord"
        tfrecords_files[split].append(tfrecord_path)


"""
Merge tfrecords inside each split into a unique tfrecord
"""
for split in splits:
    list_of_tfrecords = tfrecords_files[split]
    merged_dataset = tf.data.TFRecordDataset(list_of_tfrecords)
    dest_path = f"./output_tfrecords/{split}/merged_logos.tfrecord"
    writer = tf.data.experimental.TFRecordWriter(dest_path)
    writer.write(merged_dataset)
    len_dataset = merged_dataset.reduce(np.int64(0), lambda x, _: x + 1)
    print(f"{split} dataset contains {len_dataset} instances.") 
print("Finished merging! \n")

