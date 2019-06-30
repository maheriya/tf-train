#!/bin/csh -f
#
# A utility wrapper to save/export a model for inference
# Creates a 'frozen' graph .pb model file which includes graphs
# and parameters (TF variables converted to constants)
#
# Run this from where tensorflow/models/research or where 
# object_detection is at current dir level

## Tennis dataset; motion db
set chkpoint_basename = "train/model.ckpt-80000"
set config_file       = "train/ssdlite_mobilenet_v2_tennis.config"
set export_dir        = "exported_models/ssdlite_mobilenet_v2_tennis"

## Tennis dataset; non-motion, base color db
#set chkpoint_basename = "backup/train-tennis-2019-06-22/model.ckpt-100000"
#set config_file       = "backup/train-tennis-2019-06-22/ssdlite_mobilenet_v2_tennis.config"
#set export_dir        = "exported_models/ssdlite_mobilenet_v2_tennis_color"


## PASCAL VOC dataset
#set chkpoint_basename = "backup/train-2019-06-17/model.ckpt-100000"
#set config_file       = "backup/train-2019-06-17/ssdlite_mobilenet_v2_pascal.config"
#set export_dir        = "exported_models/ssdlite_mobilenet_v2_pascal"

set input_type="image_tensor"
python object_detection/export_inference_graph.py \
    --input_type=${input_type} \
    --pipeline_config_path=${config_file} \
    --trained_checkpoint_prefix=${chkpoint_basename} \
    --output_directory=${export_dir}

