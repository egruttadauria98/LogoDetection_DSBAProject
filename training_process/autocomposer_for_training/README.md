# Autocomposer folder
This folder contains an autocomposer notebook that is used to prepare the data for further processing in Roboflow.

The autocomposer structure:
![Alt text](/Users/elinejohansennielsen/Documents/GitHub/LogoDetection_DSBAProject/README_pictures/structure_autocomposer.png?raw=True "structure of autocomposer")

In order to run the autocomposer.ipynb the following has to be in the folder:
1. DLCV_logo_project.tar.gz: A tar.gz file containing the entire dataset provided by the professors.
2. CSV-files containing the annotations in the original dataset and noise.

The file autocomposer.ipynb begins by unzipping the tar.gz file and extracting all the data inside. The script continues to read the csv file containing the annotations provided by the professors, extracting only the logos we want and removing spaces from the class name. The edited annotations is saved to a new CSV-file. The next step is to extract only the images containing the logos we want and move them to a new folder called "data/train". After the pictures are moved, the script randomly splits the pictures into 5 equal parts and saves each part to its own folder within the folder "data/subfolders". After the script has finished, there is one folder of images for each group member. The idea is to upload these images to Roboflow in order to look over all annotations and make sure that they are good.