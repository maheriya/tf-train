#!/usr/bin/env python3
#-########################################################################################
# Load a saved TF-TRT optimized model and run inference on given input image
#-########################################################################################

from PIL import Image
import sys
import os
import urllib
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import time

t0 = time.time()
model = 'ssdlite_mobilenet_v2_coco'
INPUTS = {'MODEL'           : model,
          'MDATE'           : '2018_05_09',
          'DATA_DIR'        : './data/',
          'CONFIG_FILE'     : model + '.config',   # ./data/ssdlite_mobilenet_v2_coco.config
          'CHECKPOINT_FILE' : 'model.ckpt',    # ./data/ssdlite_mobilenet_v2_coco/model.ckpt
          'IMAGE_PATH'      : './data/huskies.jpg'}
TRT_max_GPU_mem_size = 1 << 28
TRT_precision = 'FP16' # or 'FP32' or 'INT8'
print("MODEL: {MODEL}".format(**INPUTS))
print("GPU MAX MEM for TRT: {} MB".format((TRT_max_GPU_mem_size/1024)/1024))
print("Target precision: {}".format((TRT_precision)))

#-########################################################################################
# Create TF session and load saved TF-TRT optimized model
#-########################################################################################
# Inference with TF-TRT frozen graph workflow:
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
#         # First deserialize your frozen graph:
#         with tf.gfile.GFile("{DATA_DIR}/{MODEL}_frozen.pb".format(**INPUTS), 'rb') as f:
#             graph_def = tf.GraphDef()
#             graph_def.ParseFromString(f.read())
# Now you can create a TensorRT inference graph from your frozen graph:
t1 = time.time()
print("Created session. [Time spent: {:.2f} secs]".format(t1 - t0))
t0 = t1
print("Now reading the frozen model graph.")
with tf.gfile.GFile("{DATA_DIR}/{MODEL}_trt.pb".format(**INPUTS), 'rb') as f:
    trt_graph = tf.GraphDef()
    trt_graph.ParseFromString(f.read())

tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')


t1 = time.time()
print("Done loading graph. [Time spent: {:.2f} secs]".format(t1 - t0))
#-########################################################################################
# Following two tasks 1 and 2 could be a loop
#-########################################################################################
#-########################################################################################
# 1. Load image
image = Image.open(INPUTS['IMAGE_PATH'])
image_resized = np.array(image.resize((300, 300)))
image = np.array(image)
#-########################################################################################
# 2. Run inference
t0 = time.time()
_scores, _boxes, _classes, _num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], 
                                                          feed_dict={tf_input: image_resized[None, ...]})
t1 = time.time()
print("Detection on one image done. [Time spent: {:.2f} secs]".format(t1 - t0))



#-########################################################################################
# For debug: Save annotated image for viewing
#-########################################################################################
boxes          = _boxes[0] # index by 0 to remove batch dimension
scores         = _scores[0]
classes        = _classes[0]
num_detections = int(_num_detections[0])
print("Num detections: ", num_detections)
print("Scores:", scores[0:5])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(image)

#fig, ax = plt.subplots(1,1)
#plt.axis('off')
print("Image size:", image.shape)
for i in range(num_detections):
    # scale box to image coordinates
    box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])

    # display rectangle
    ## TBD : Use OpenCV to draw boxes instead of patches.
    patch = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], color='g', alpha=0.3)
    ax.add_patch(patch)

    # display class index and score
    plt.text(x=box[1] + 10, y=box[2] - 10, s='%d (%0.2f) ' % (classes[i], scores[i]), color='w')
    
fig.savefig('output.jpg')

#-########################################################################################
## Bench mark. Better with different images. For now, just use one image
## This is a best case result due to cached image
#-########################################################################################
num_samples = 200

t0 = time.time()
for i in range(num_samples):
    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
        tf_input: image_resized[None, ...]
    })
t1 = time.time()
print('Average runtime: %f seconds' % (float(t1 - t0) / num_samples))



#EOF
