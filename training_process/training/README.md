# Training Folder

The training folder is the place in which the building process of the model takes place, starting from the download of the pre-trained weights and ending with the storage of the results of the fine-tuning procedure. 

The central part of the training folder is the training.ipynb Jupyter Notebook, that contains the whole pipeline for downloading the pre-trained weights, the processing of the dataset as an input for the fine-tuning process, the evaluation of the model after training and the saving of such fine-tuned model. 

In the most basic setting, the training folder should contain the Jupyter Notebook _training.ipynb_, the *pictures* folder (containing itself the *output_tfrecord* folder that has the train and validation TFRecords in it) and the file _ModelSettings.py_, in which a function is defined and called inside the training.ipynb to gather all the possible model structures offered in Tensorflow's 2 Object Detection API ModelZoo.

We decided to use Tensorflow because a sample Colab Notebook was available on the internet and, while reading through it, we found out that the process was pretty clear, intuitive, and could deliver potentially good results by constructing a well-defined hierarchical pattern.

In the file training.ipynb the following happens:
1. The *model* folder is cloned from TF2 Object Detection API. This cloning of a git repo is important to make the model run. 
2. To make the model folder usable, some parameters need to be manually changed, as indicated in the markdown cells following the download. 
3. The *pycoco* folder is cloned from TF2. Once again, the folder needs to be tweaked (with minor changes) to perform the training. These minor changes of major importance avoid collision of packages that hasn't still been resolved in terms of documentation.
4. The input data are imported from the 