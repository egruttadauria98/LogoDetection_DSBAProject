# Preprocessing

The goal of our preprocessing pipeline is to take the pictures we manually annotated on Roboflow and produce two .tfrecord files: one for the training and one for the validation data. These two files are then used to train the model. 

We do not produce a .tfrecord for the test data, since the evaluation is ran on a .csv file and a folder containing the test pictures. In fact, before even beginning the preprocessing, we take the test data and move it to the *INFERENCE_DIR* folder, inside *inference_PROD*. This way we make sure that the test data is left untouched during the preprocessing and that no bias is being introduced!

Let's analyze the preprocessing steps in detail.

## 1. Roboflow's output

Once we annotated all pictures manually using Roboflow, each of us 6 exported a chunk of the dataset in Tensorflow Object Detection format (each chunk containing around 7000 pictures). This format produces a folder containing three sub-folders (one for train, validation and test respectively), each contaning the pictures corresponding to that split and a .csv file contaning the annotations of those pictures.

The reason we decided to export our pictures in this format instead of exporting directly into TFRecord format, is that we needed more control in the preprocessing steps and working with .csv files is less error-prone than working on .tfrecords.

Hence, at the beginning of the preprocessing step, all folders should be empty, except for the *data* folder. The latter should contain 6 zipped folder, each containing the Roboflow's output of one group member (e.g. Batch_Eline, Batch_AleL, etc.).

## 2. Annotations cleaning

The next step in the process is running the *clean_annotations.py* script, which does the following:
1. Unzip the 6 folders in data
2. For each of the 6 unzipped folders and for each of the three splits (train, valid, test): remove all the pictures having logos different from the ones we want to detect. This is done both by removing the pictures from the .csv annotations file and by physically removing the pictures from the folder.

At the end of this step, the *data* folder should contain 6 unzipped folders, still in Tensorflow Object Detection format, each containing one sub-folder for each dataset split. These folders contain the original *_annotations.csv* file, a new *_annotations_clean.csv* file and the remaining pictures.

## 3. Class balancing

Since our classes were quite unbalanced (e.g. Nike was over-represented, whereas something like The North Face was under-represented), we apply this step to balance the classes. In particular, we identify 4 logo macro-groups, each corresponding to a frequency bracket (Example: Logo Group 1 contains Nike, whereas Logo Group 4 contains the most under-represented logos). Different groups are associated with different augmentation. That is, a small degree of augmentation is done on over-represented logos, and a large degree on under-represented logos.

The *balance_classes.py* script takes the 6 clean annotations .csv files produced in the previous step and produces a big, single annotations file. It then creates a folder called *data_balanced_folder* which contains one folder for each of the 12 logos we decided to include. It then moves pictures corresponding to those logos from the *data* folder to each logo's folder inside *data_balanced_folder*.

At the end, we used the 12 logo folders to manually produce 4 Logo Group folders, using the reasoning explained above. The reason why we decided to do so manually is that not doing so created some inconsistencies inside the *logos_label_map.pbtxt* file. This file associates logos name to id's and is fundamental for the training process when dealing with TFRecord files.

## 4. Roboflow - Part 2

We then took each of the 4 Logo Group folders and uploaded it in a different Roboflow project. The reason why we did so is that this allowed us to apply different augmentation to different groups. Hence, we applied much more augmentation on under-represented logos and not as much on over-represented logos. 

Finally, we exported each of the 4 projects in TFRecord format. 

## 5. Merging the TFRecords

At the end of the previous step, we exported 4 folders in TFRecord format. Each of these contains two splits, one for training and one for validation (remember: we left test out in the beginning and never used it in the preprocessing). Each of these two splits, contains a .tfrecord file and a .pbtxt file. The former is our dataset's file, the latter is a mapping between logo id's and names, which is necessary when working with .tfrecord files.

The goal of this step is to merge the 4 .tfrecord files into a single one to use during training. The label map is the same across the 4 groups and we will use the same label map with the larger .tfrecord file.

To merge the .tfrecord files we use the *merge_tfrecords.py* script, which does the following:
- Unzips the 4 folders containing the data in TFRecord format
- Creates a folder named *output_tfrecords* which will contain the merged .tfrecord files and the label map
- Moves the *logos_label_map.pbtxt* file to the *output_tfrecords* folder.
- Merges the four .tfrecord files into a single one

The steps above are applied for both train and validation data, so in the end the *output_tfrecords* folder will contain two sub-folders (one for each split), each containing the merged .tfrecord file for that split and the label map.

These output files are then used to train the model.