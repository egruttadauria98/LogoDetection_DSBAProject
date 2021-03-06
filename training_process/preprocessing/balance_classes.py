import os
import shutil
import pandas as pd

"""
Merge each _clean_annotations.csv file into a single big dataframe
"""


def move_images(current_dir, dest_dir):
    for element in os.listdir(current_dir):
        if element.endswith(".jpg"):
            shutil.move(os.path.join(current_dir, element), os.path.join(dest_dir, element))

    print(f"Moving images from {current_dir} to {dest_dir}")


train_destination_dir = os.path.join(os.getcwd(), "data\\train_merged")

if "train_merged" in os.listdir("data"):
    os.rmdir("./data/train_merged")
elif "train_merged" not in os.listdir("data"):
    os.mkdir(train_destination_dir)

    batches = ["BatchAleC", "BatchAleL", "BatchEline", "BatchElio", "BatchReb"]
    splits = ["train", "valid"]

    path_first_trainfolder = f"./data/{batches[0]}/{splits[0]}"
    path_first_validfolder = f"./data/{batches[0]}/{splits[1]}"

    # Initialize a dataframe as the merged csv files of train and valid for the first batch
    df = pd.concat([pd.read_csv(f"{path_first_trainfolder}/_annotations_clean.csv"),
                    pd.read_csv(f"{path_first_validfolder}/_annotations_clean.csv")])

    move_images(path_first_trainfolder, train_destination_dir)
    move_images(path_first_validfolder, train_destination_dir)

    # Merge remaining batches to the df
    for batch in batches[1:]:
        for split in splits:
            origin_folder = f"./data/{batch}/{split}"
            df = pd.concat([df, pd.read_csv(f"{origin_folder}/_annotations_clean.csv")])
            move_images(origin_folder, train_destination_dir)

    df.to_csv(os.path.join(train_destination_dir, "merged_annotations_clean.csv"), index=False)

    print(f"Files in train merged are {len(os.listdir('./data/train_merged')) - 1}")

"""
Identify groups
"""

data_trainmerged = "./data/train_merged"
balanced_destination_dir = "./data_balanced_folder"

if "data_balanced_folder" in os.listdir(os.getcwd()):
    os.rmdir("data_balanced_folder")
    os.mkdir(balanced_destination_dir)
elif "data_balanced_folder" not in os.listdir(os.getcwd()):
    os.mkdir(balanced_destination_dir)

original_df = pd.read_csv(f"{data_trainmerged}/merged_annotations_clean.csv")

logo_groups = [["TheNorthFace"], ["Puma"], ["UnderArmour"], ["AppleInc"], ["MercedesBenz"], ["NFL"],
               ["Starbucks"], ["Adidas"], ["Nike"], ["Emirates"], ["CocaCola"], ["HardRockCafe"]]

for group in logo_groups:
    df_group = original_df[original_df["class"].isin(group)]
    image_names_pergroup = df_group["filename"].unique()

    df_image_names = original_df[original_df["filename"].isin(image_names_pergroup)]

    balanced_splitted_dir = f"{balanced_destination_dir}/{group[0]}"

    if group[0] not in os.listdir(balanced_destination_dir):
        os.mkdir(balanced_splitted_dir)
    else:
        os.rmdir(balanced_splitted_dir)
        os.mkdir(balanced_splitted_dir)

    for image in df_image_names["filename"].unique():
        shutil.move(f"{data_trainmerged}/{image}", f"{balanced_splitted_dir}/{image}")

    print(f"Amount of images in {balanced_splitted_dir} is {len(os.listdir(balanced_splitted_dir)) - 1}")

    original_df = original_df[~original_df["filename"].isin(image_names_pergroup)]

"""

logo_to_folder = {"Nike": "Nike", "Adidas": "Adidas", "MercedesBenz": "MercedesBenz",
                  "Starbucks": "MercedesBenz", "AppleInc": "AppleInc", "Puma": "AppleInc",
                  "UnderArmour": "AppleInc", "TheNorthFace": "AppleInc", "NFL": "AppleInc"}

folders = ["Nike", "Adidas", "MercedesBenz", "AppleInc"]

balanced_splitted_dir = "data_balanced_folder"

images_already_moved = []
for folder in folders:
    for logo in logo_to_folder.keys():
        # if the logo is already in the folder, ignore it
        if logo_to_folder[logo] == folder:
            continue
        logo_df = df[df["class"] == logo]
        images_to_move = logo_df[~logo_df["filename"].isin(images_already_moved)]["filename"].to_list()
        # images_already_moved += images_to_move
        i = 0
        while i != 100:
            image = images_to_move[i]
            try:
                img_path = f"{balanced_splitted_dir}/{logo_to_folder[logo]}/{image}"
                dest_path = f"{balanced_splitted_dir}/{folder}/{image}"
                shutil.move(img_path, dest_path)
                i += 1
                images_already_moved.append(image)
            except:
                pass
"""
