# Training Folder

The training folder is the place in which the building process of the model takes place, starting from the download of the pre-trained weights and ending with the storage of the results of the fine-tuning procedure. 

The central part of the training folder is the training.ipynb Jupyter Notebook, that contains the whole pipeline for downloading the pre-trained weights, the processing of the dataset as an input for the fine-tuning process, the evaluation of the model after training and the saving of such fine-tuned model. 

In the most basic setting, the training folder should contain the Jupyter Notebook _training.ipynb_, the *pictures* folder (containing itself the *output_tfrecord* folder that has the train and validation TFRecords in it) and the file _ModelSettings.py_, in which a function is defined and called inside the training.ipynb to gather all the possible model structures offered in Tensorflow's 2 Object Detection API ModelZoo.

We decided to use Tensorflow because a sample Colab Notebook was available on the internet and, while reading through it, we found out that the process was pretty clear, intuitive, and could deliver potentially good results by constructing a well-defined hierarchical pattern.

In the file training.ipynb the following happens:
1. The *model* folder is cloned from TF2 Object Detection API. This cloning of a git repo is important to make the model run. 
2. To make the model folder usable, some parameters need to be manually changed, as indicated in the markdown cells following the download. 
3. The *pycoco* folder is cloned from TF2. Once again, the folder needs to be tweaked (with minor changes) to perform the training. These minor changes of major importance avoid collision of packages that hasn't still been resolved in terms of documentation.
4. The input data are imported from the *pictures* folder. Data consistency and data format is, in this step, extremely important. If the TFRecords haven't been created rigorously, the model might not get the right inputs and might end up learning badly or even outputting some error. 
5. To train a TF2 Model, another important step is that of collecting the pre-trained weights, and change the configuration file of the model accordingly, to decide for the batch size and number of evaluation steps, for example. 
    * To be able to use different models, we populated the file ModelSetting.py with the models we thought would be good to train. 
    * From the ModelZoo, it is possible to pick the pre_trained_checkpoints (extension tar.gz) and the model_name (that must be the same as the one given to the pre_trained_checkpoints): https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md.By entering the *raw* version of the ModelZoo page on GitHub, it is possible to pick the the names of the tar.gz.
    * To find instead the configurations: https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2
6. Once the configuration file is changed, the folder **TENSOR_RESULTS** is generated and in there all the checkpoints in the training are stored. 
7. After the training is done (or stopped, if the model is overfitting or not learning anymore, as it's possible to see from the Tensorboard) evaluation steps are performed. 
8. Once evaluation has taken place, the static graph of the now fine-tuned model is picked and stored inside the newly-generated **FINE_TUNED_MODEL**. This allows us to retrieve the final model version at a later step, above all when doing IoU computation and inference inside the inference directory.


## How to reproduce our model history?
At this step, the training paths inserted inside the Jupyter Notebook are highly specific to the virtual machine, as it was the only machinery we used to train our model, having an integrated TeslaK80 GPU. 
We didn't change the paths mostly because it wouldn't have made sense to create a local-friendly version of the training process, though it still remains highly reproducible if ever needed.

We will be delivering all of the missing folders, though. In fact, you will be provided with:
* The *pictures* folder, containing the folder *output_tfrecords*
* The *models* folder, ready and changed
* The *pycoco* folder
* Our *TENSOR_RESULTS* folder, having all of our trained models
* Our *FINE_TUNED_MODEL* folder, having all of our saved models (and for this, the structure is the same as the *TENSOR_RESULTS* folder of course)

**Copy and paste these folders from the file we shared to this section, and you could see our models evolution!**
