#!/bin/csh -f
##
## Wrapper script to convert VOC dataset to TFRecord
python3 object_detection/dataset_tools/create_pascal_tf_record.py \
  --label_map_path=object_detection/data/pascal_label_map.pbtxt \
  --data_dir=VOCdevkit --year=VOC2012 --set=train \
  --output_path=TFRecord/pascal_train.record

python3 object_detection/dataset_tools/create_pascal_tf_record.py \
  --label_map_path=object_detection/data/pascal_label_map.pbtxt \
  --data_dir=VOCdevkit --year=VOC2012 --set=val \
  --output_path=TFRecord/pascal_val.record

