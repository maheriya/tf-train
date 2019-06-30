#!/usr/bin/env python
#-########################################################################################
# Load a saved TF model frozen inference graph and run inference on given input images
#-########################################################################################

from __future__ import print_function
#from PIL import Image
import cv2 as cv
import sys
import os
import subprocess
import time
import re
from glob import glob
import argparse
import numpy as np
import tensorflow as tf
from object_detection.utils.visualization_utils import draw_bounding_boxes_on_image

if sys.version_info[0] < 3:
    PYVER = 2
else:
    PYVER = 3




'''
This clasee represents a VOC dataset that contains a collection of sequential video frames
The dataset may or may not be annotated. If annotated ('Annotations/*.xml exist), then
certain methods -- like evaluation against ground truth -- become available to caller.
This is the base class. Based on this, there will be another class which will handle motion
information.
'''
class VOCVideoDataset:
    def __init__(self, vocbase, out_imgsize=(480, 270), dbtype='TENNIS', motion=True): 
        '''
        vocbase: VOC base (or 'year') directory that contains dataset
        '''
        self.out_imgsize = out_imgsize
        self.initdone = False
        self.dbtype   = dbtype
        self.motion_db= motion
        self.imgdir = os.path.join(vocbase, "JPEGImages")
        self.anndir = os.path.join(vocbase, "Annotations")
        print("Processing images from {}".format(self.imgdir))

        if self.dbtype == 'TENNIS':
            ## Tennis Dataset labels
            self.LBL_NAMES = [ "__bg__", "ball", "racket", "otherball" ]
        elif self.dbtype == 'VOC':
            ## Classic PASCAL VOC labels
            self.LBL_NAMES = [ "__bg__",
                    "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cow",
                    "DiningTable", "Dog", "Horse", "Motorbike", "Person", "Pottedplant", "Sheep", "Sofa",
                    "Train", "TVMonitor" ]
        else: # Assume COCO labels
            self.LBL_NAMES = [ "__bg__",
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
                "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]
 
        self.LBL_COLORS = [
                (0xde, 0xde, 0xde), # background
                (0x48, 0x0c, 0xb8), # id 1
                (0x53, 0xb8, 0x09), # id 2
                (0xb8, 0x84, 0x0c), # ...
                (0x48, 0x0c, 0xb8),
                (0x53, 0xb8, 0x09),
                (0xb8, 0x84, 0x0c),
                (0x48, 0x0c, 0xb8),
                (0x53, 0xb8, 0x09),
                (0xb8, 0x84, 0x0c),
                (0x48, 0x0c, 0xb8),
                (0x48, 0x0c, 0xb8),
                (0x53, 0xb8, 0x09),
                (0xb8, 0x84, 0x0c),
                (0x48, 0x0c, 0xb8),
                (0x53, 0xb8, 0x09),
                (0xb8, 0x84, 0x0c),
                (0x53, 0xb8, 0x09),
                (0xb8, 0x84, 0x0c),
                (0x48, 0x0c, 0xb8),
                (0x53, 0xb8, 0x09)]
        if self.dbtype == 'COCO':
            self.LBL_COLORS.append(self.LBL_COLORS) 
            self.LBL_COLORS.append(self.LBL_COLORS) 
            self.LBL_COLORS.append(self.LBL_COLORS) 



    def getNumberingScheme(self, imgname):
        fnum     = re.sub(r'.*[-_](\d+).jpg', r'\1', imgname)
        fpre     = re.sub(r'(.*[-_])(\d+).jpg', r'\1', imgname)
        numlen   = len(fnum)
        numtmplt = '{:0' + str(numlen) + 'd}'
        return (fpre, numtmplt)

    def collectImages(self):
        '''Collect all images in an array and prepare for processing'''
        imgs = glob("{}/*.jpg".format(self.imgdir))
        imgs = [os.path.basename(i) for i in imgs]
        imgs.sort() # Sort images to pick frames in order. It is assumed the images are named likewise
    
        (self.fprefix, self.ntemplate) = self.getNumberingScheme(imgs[0])
        print("fprefix: {}, template: {}".format(self.fprefix, self.ntemplate))
        self.images = imgs
        self.imgbase = os.path.splitext(os.path.basename(imgs[0]))[0]
        if self.motion_db:
            self.index = 2  ## Index of current image. For motion db, this starts at 2
        else:
            self.index = 0  ## Index of current image. For motion db, this starts at 2
        self.initdone = True

    def getNextImage(self):
        '''
        Each call will return one image
        Returns success/fail status and image
        '''
        if not self.initdone:
            self.collectImages()

        imgs = self.images
        if self.index >= len(imgs):
            return (False, None)

        img_i  = imgs[self.index]
        if self.motion_db:
            img_i1 = imgs[self.index-1]
            img_i2 = imgs[self.index-2]

        self.index += 1
    
        if self.motion_db:
            fnum    = int(re.sub(r'.*[-_](\d+).jpg', r'\1', img_i))
            eimg_i  = self.fprefix + self.ntemplate.format(fnum) + '.jpg'
            eimg_i1 = self.fprefix + self.ntemplate.format(fnum-1) + '.jpg'
            eimg_i2 = self.fprefix + self.ntemplate.format(fnum-2) + '.jpg'
            if img_i != eimg_i or img_i1 != eimg_i1 or img_i2 != eimg_i2:
                # Not a continuous series of three frames including previous two, we skip this frame
                print("Skipping {}".format(img_i))
                self.getNextImage()
    
    
        if not self.motion_db:
            ## load images as color
            cvimg_n  = cv.imread(os.path.join(self.imgdir, img_i), 1)
        else:
            ## load images as grayscale
            cvimg  = cv.imread(os.path.join(self.imgdir, img_i), 0)
            cvimg1 = cv.imread(os.path.join(self.imgdir, img_i1), 0)
            cvimg2 = cv.imread(os.path.join(self.imgdir, img_i2), 0)
            ## Combine
            cvimg_n  = cv.merge((cvimg, cvimg1, cvimg2))


        return (True, cvimg_n)

    def writeBBLabelText(self, img, p1, p2, lblname, lblcolor, lblconf=None):
        '''
        img: image array
        p1, p2: bounding box upper-left and bottom-right points
        lblname: label name
        lblcolor: color tuple
        '''
        fontFace = cv.FONT_HERSHEY_PLAIN
        fontScale = 1.
        thickness = 2
        lblname += ' {:3.0f}%'.format(lblconf*100)
        textSize, baseLine = cv.getTextSize(lblname, fontFace, fontScale, thickness)
        txtRBx = p1[0] + textSize[0] + 2
        txtRBy = p1[1] + textSize[1] + 2
    
        img = cv.rectangle(img, p1, (txtRBx, txtRBy), lblcolor, cv.FILLED)
        textOrg = (p1[0]+thickness, p1[1]+textSize[1])
        img = cv.putText(img, lblname,
                         textOrg, fontFace, fontScale,
                         (255,255,255),      # inversecolor = Scalar::all(255)-lblcolor
                         thickness)
        return img


    def drawBoundingBox(self, image, bbox, lblid, lblconf=None):
        """Draw bounding box on the image."""
        ymin,xmin,ymax,xmax = bbox
        lblcolor = self.LBL_COLORS[lblid]
        lblname  = self.LBL_NAMES[lblid]
        image = cv.rectangle(image, (xmin, ymin), (xmax, ymax), lblcolor, 2)
        image = self.writeBBLabelText(image, (xmin, ymin), (xmax, ymax), lblname, lblcolor, lblconf)
        return image



def doInference(vocbase, dbtype, motion_db):
    if dbtype == 'TENNIS':
        if motion_db:
            ## Motion model; trained with pre-scaled 480x270 images
            SSD_MODEL = 'exported_models/ssdlite_mobilenet_v2_tennis/frozen_inference_graph.pb'
        else:
            ## Base model (color), trained with 1920x1080 images rescaled at run time to 480x270
            SSD_MODEL = 'exported_models/ssdlite_mobilenet_v2_tennis_color/frozen_inference_graph.pb'
        IMGSIZE   = (480, 270)
    elif dbtype == 'VOC':
        ## Our own PASCAL VOC data trained model.
        SSD_MODEL = 'exported_models/ssdlite_mobilenet_v2_pascal/frozen_inference_graph.pb'
        IMGSIZE   = (300, 300)
    else: # Assume COCO
        ## Pretrained ms-coco model
        SSD_MODEL = 'tf-train/models/ssdlite_mobilenet_v2_coco/frozen_inference_graph.pb'
        IMGSIZE   = (300, 300)


    t0 = time.time()
    vdb = VOCVideoDataset(vocbase, IMGSIZE, dbtype, motion_db)
    
    #-########################################################################################
    # Create TF session and load saved model
    #-########################################################################################
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    t1 = time.time()
    print("Created session. [Time spent: {:.2f} secs]".format(t1 - t0))
    t0 = t1
    print("Now reading the frozen model graph.")
    frozen_graph = tf.GraphDef()
    with tf.gfile.GFile(SSD_MODEL, 'rb') as f:
        frozen_graph.ParseFromString(f.read())
    
    tf.import_graph_def(frozen_graph, name='')
    
    tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')
    
    t1 = time.time()
    print("Done loading graph. [Time spent: {:.2f} secs]".format(t1 - t0))
    while True:
        # 1. Load image
        st, cvimage = vdb.getNextImage()
        if st == False:
            print("All images are processed")
            break
        image_resized = cv.resize(cvimage, (480, 270), interpolation = cv.INTER_CUBIC)
    
        # 2. Run inference
        t0 = time.time()
        _scores, _boxes, _classes, _num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], 
                                                                  feed_dict={tf_input: image_resized[None, ...]})
        t1 = time.time()
        print("Detection on one image done. [Inference latency: {:.0f} ms]".format((t1 - t0)*1000))
    
        #-########################################################################################
        # For debug: Save annotated image for viewing
        #-########################################################################################
        boxes          = _boxes[0] # index by 0 to remove batch dimension
        scores         = _scores[0]
        classes        = _classes[0]
        num_detections = int(_num_detections[0])
        print("Num detections: ", num_detections)
        #print("Scores:", scores)
        #print("Boxes:", boxes)
        #print("Classes:", classes)
        height         = cvimage.shape[0]
        width          = cvimage.shape[1]

        for i in range(num_detections):
            if (scores[i] > 0.35):
                bbox = [int(b) for b in (boxes[i] * np.array([height, width, height, width]))]
                lblid = int(classes[i])
                cvimage = vdb.drawBoundingBox(cvimage, bbox, lblid, scores[i])  ## t-l row,col, b-r row,col
        cv.imshow("Image", cvimage)
        key = cv.waitKey(0) & 255
        if key == 27:
            break
    cv.destroyAllWindows()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "vocbase", type=str, default="/IMAGESETS/TENNIS/VOCdevkit/IMX274-b667e241",
        help="The VOC base (or 'year') directory."
    )
    parser.add_argument(
        "--type", dest='dbtype', type=str, default="TENNIS",
        help="Type of lable/database. 'VOC', 'COCO' or 'TENNIS'. Used for model selection as well as label display."
    )
    parser.add_argument(
        '--no-motion', dest='motion_db', action='store_false', required=False,
        help='Do not use tennis motion database (use basic non-motion color db).'
    )
    parser.add_argument(
        '--motion', dest='motion_db', action='store_true', required=False,
        help='Use tennis motion database (use basic non-motion color db). (default)'
    )
    parser.set_defaults(motion_db=True)
    args = parser.parse_args()
    doInference(args.vocbase, args.dbtype, args.motion_db)
    print("Done!")

#EOF
