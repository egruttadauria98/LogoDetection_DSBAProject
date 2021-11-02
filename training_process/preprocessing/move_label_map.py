import os
import shutil

source_file = "./logos_label_map.pbtxt"
splits = ["train", "valid", "test"]

for folder in os.listdir("./tfrecords"):
    for split in splits:
        dest_dir = f"./tfrecords/{folder}/{split}"
        shutil.copy(source_file, dest_dir)

print("Label map successfully moved to tfrecord folders! \n")
