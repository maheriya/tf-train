# TensorFlow Models Training
A collection of scripts to train TensorFlow models.

These scripts are created to support datasets that are originally in Pascal VOC format.
However, it is possible to use other object detection dataset formats as long as we can 
convert them into TFRecord format.



## Required Installations
Install TensorFlow by following [official TensorFlow installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). TensorFlow Object Detection API is used for training and evaluation in the scripts.

## Set up Training Work Space
TBD 
This repository contains TensorFlow model config for MobileNetV2 and scripts to train and run inference. However, certain directory structure, etc. is expected. I will document this as I find more time.


## Prepare Data
Refer to [Preparing Inputs page]() for reference. I will open up more scripts that are currently in my private tennisLabels repository once I comb them to remove sensitive information if any. I wrote them to prep the dataset from ground up. These script allow you to create training datasets in TFRecords format starting with input videos, labelling using CVAT including conversion to Pascal VOC, creating motion dataset, and finally, converting them to TFRecords. However, for this repo, I am assuming you have your dataset already in TFRecords format.

## Usage

### Training
Once the dataset and config are ready, simply execute the [utils/run_training](utils/run_training) shell script to launch the training. Of course, you will have to modify it to match your directory and config file paths.

Once training starts, you can monitor the training progress using TensorBoard. \[TBD: how to monitor \]

### Export model
Once training completes, export the model using [utils/export_for_inference.csh](utils/export_for_inference.csh) script. This will create a model that is ready for inference.

### Run Inference
Using the inference model exported in the step above, you can run inference using [utils/tf_inference.py](utils/tf_inference.py) script. There is also a [utils/trt_inference.py](utils/trt_inference.py) script if you have an Nvidia GPU, and TensorRT installed. The latter will optimize the model to accelerate it using TennsorRT.





