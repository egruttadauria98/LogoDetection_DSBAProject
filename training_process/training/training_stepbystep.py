# -*- coding: utf-8 -*-

"""
We will take the following steps to implement YOLOv4 on our custom data:
* Install TensorFlow2 Object Detection Dependencies
* Download Custom TensorFlow2 Object Detection Dataset
* Write Custom TensorFlow2 Object Detection Training Configuation
* Train Custom TensorFlow2 Object Detection Model
* Export Custom TensorFlow2 Object Detection Weights
* Use Trained TensorFlow2 Object Detection For Inference on Test Images
"""

import os
import pathlib
from git import Repo
import logging

logging.basicConfig(level=logging.INFO)

models_folder = os.path.join(os.getcwd(), "models")

# Clone the tensorflow models repository if it doesn't already exist in this folder
if "models" in pathlib.Path.cwd().parts:
    while "models" in pathlib.Path.cwd().parts:
        os.chdir('..')

    logging.info("The models have already been uploaded. Change working directory to the models folder.")

elif not pathlib.Path('models').exists():
    os.mkdir("./models")
    repo = Repo.clone_from(
        'http://RebSolcia:Clementinabookie18121998!@github.com/tensorflow/models.git',
        models_folder,
        depth=1,
        branch='master',
    )

    logging.info("The models have now been loaded from the tensorflow/models.git repo.")

# Commented out IPython magic to ensure Python compatibility.
# # Install the Object Detection API
