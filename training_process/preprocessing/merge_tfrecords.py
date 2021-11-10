"""
This script takes tfrecords from ./tfrecords and merge them into a single folder inside ./output_tfrecords
"""

import os
import shutil
import tensorflow as tf
import numpy as np
import zipfile

"""
Unzip tfrecords in ./tfrecords
"""
src_dir = "./tfrecords/"
for fname in os.listdir(src_dir):
    if not "train" in fname:
            continue
    print(f"Unzipping {fname}")
    zipfile_path = src_dir + fname
    new_name = fname.split(".")[0]
    dest_dir = src_dir + new_name
    try:
        os.mkdir(dest_dir)
    except:
        pass
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
print("Finished unzipping! \n")

"""
Move label map to ./output_tfrecords
"""
dest_path = "./output_tfrecords/train"
#source_file = "./logos_label_map.pbtxt"
#try:
#    os.makedirs(dest_path, exist_ok=True)
#except:
#    pass
#shutil.copy(source_file, dest_path)

"""
Create list of tfrecord files for each split
"""
tfrecords_files = []
for folder in os.listdir("./tfrecords"):
    if ".zip" not in folder:
        if "README" not in folder:
            tfrecords_files.append(f"./tfrecords/{folder}/train/logo.tfrecord")
        else:
            print("It is not a zip but it's a README!")
    else:
        print("It is a zip")


"""
Merge tfrecords into a unique one
"""
merged_dataset = tf.data.TFRecordDataset(tfrecords_files)
dest_path = f"./output_tfrecords/train/merged_logos.tfrecord"
writer = tf.data.experimental.TFRecordWriter(dest_path)
writer.write(merged_dataset)
len_dataset = merged_dataset.reduce(np.int64(0), lambda x, _: x + 1)
print(f"Train dataset contains {len_dataset} instances.") 
print("Finished merging! \n")