"""
This script does the following:
    1. Unzip the datasets in the data folder
    2. Clean the annotation csv file of each split by: removing empty rows, removing Ralph Laurent, Intimissimi 
       and Null labels
    3. Clean the folder containing the images of each split removing the pictures that are not 
       contained in the cleaned annotation csv file
"""

import pandas as pd
import zipfile
import os

# 1. Unzip datasets in data
src_dir = "./data/"
for fname in os.listdir(src_dir):
    if not "Batch" in fname:
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

# 2. Clean the annotation csv file of each split and 3. Clean each split's folder by removing 
# images that do not appear in the clean annotations csv
def clean_annotations(annotations_filepath):
    """
    Given the filepath of an annatations csv file, create a clean version of the csv file
    """
    # Remove Ralph Laurent, Intimissimi and null classes

    df = pd.read_csv(annotations_filepath)
    logos_to_include = ["Nike", "Adidas", "UnderArmour", "Puma", "TheNorthFace"]
    df = df[df["class"].isin(logos_to_include)]
    df["xmin"] = df["xmin"] / df["width"]
    df["xmax"] = df["xmax"] / df["width"]
    df["ymin"] = df["ymin"] / df["height"]
    df["ymax"] = df["ymax"] / df["height"]
    print(df.dtypes)
    output_csv_path = annotations_filepath[:-16] + "_annotations_clean.csv"
    df.to_csv(output_csv_path, index=False)

splits = ["train", "valid", "test"]

for fname in os.listdir("./data/"):
    if fname[-4:] == ".zip" or not "Batch" in fname:
        continue
    for split in splits:
        annotations_filepath = f"./data/{fname}/{split}/_annotations.csv"
        # Clean annotations csv file
        print(f"Cleaning annotations csv file in {annotations_filepath[:-17]}...")
        clean_annotations(annotations_filepath)
        # Delete images that do not appear in the clean csv file
        print(f"Removing pictures not appearing in the csv file...")
        unique_pics = pd.read_csv(annotations_filepath[:-16] + "_annotations_clean.csv")["filename"].unique()
        for img in os.listdir(f"./data/{fname}/{split}"):
            if img[-4:] == ".jpg" and img not in unique_pics:
                img_path = f"./data/{fname}/{split}/{img}"
                os.remove(img_path)
    print(f"Finished cleaning ./data/{fname}\n")
print("Cleaning complete! \n")



