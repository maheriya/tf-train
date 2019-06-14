#-##########################################################################
# View TFRecord created for a VOC Pascal object detection database.
# As long as the VOC Pascal format was used for creating the TFRecord,
# this script will be work.
#
# Usage:
#    python3 view_record.py --record=data.record
#-##########################################################################
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

FLAGS = None
IMG_SIZE = 112
MARK_SIZE = 68 * 2
POSE_SIZE = 3

LBL_NAMES = [
        "__bg__", "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus",
        "Car", "Cat", "Chair", "Cow", "DiningTable", "Dog", "Horse", "Motorbike",
        "Person", "Pottedplant", "Sheep", "Sofa", "Train", "TVMonitor"]

LBL_COLORS = [
      (0xde, 0xde, 0xde), # background
      (0x48, 0x0c, 0xb8), # Aeroplane
      (0x53, 0xb8, 0x09), # ...
      (0xb8, 0x84, 0x0c),
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
      (0x53, 0xb8, 0x09)] # TVMonitor


def writeBBLabelText(img, p1, p2, lblname, lblcolor, lblconf=None):
    '''
    img: image array
    p1, p2: bounding box upper-left and bottom-right points
    lblname: label name
    lblcolor: color tuple
    '''
    fontFace = cv.FONT_HERSHEY_PLAIN
    fontScale = 1.
    thickness = 2
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


def drawBoundingBox(image, bbox, lblid):
    """Draw bounding box on the image."""
    ymin,xmin,ymax,xmax = bbox
    lblcolor = LBL_COLORS[lblid]
    lblname  = LBL_NAMES[lblid]

    image = cv.rectangle(image, (xmin, ymin), (xmax, ymax), lblcolor, 2)
    image = writeBBLabelText(image, (xmin, ymin), (xmax, ymax), lblname, lblcolor)
    return image



def parse_tfrecord(filenames):
    """Try to extract a image from the record file as jpg file."""
    dataset = tf.data.TFRecordDataset(filenames)

    # Create a dictionary describing the features. This dict should be
    # consistent with the one used while generating the record file.
    feature_description = {
      'image/height':     tf.FixedLenFeature([], tf.int64),
      'image/width':      tf.FixedLenFeature([], tf.int64),
      'image/filename':   tf.FixedLenFeature([], tf.string),
      'image/source_id':  tf.FixedLenFeature([], tf.string),
      'image/key/sha256': tf.FixedLenFeature([], tf.string),
      'image/encoded':    tf.FixedLenFeature([], tf.string),
      'image/format':     tf.FixedLenFeature([], tf.string),
      'image/object/bbox/xmin':   tf.VarLenFeature(tf.float32),
      'image/object/bbox/xmax':   tf.VarLenFeature(tf.float32),
      'image/object/bbox/ymin':   tf.VarLenFeature(tf.float32),
      'image/object/bbox/ymax':   tf.VarLenFeature(tf.float32),
      'image/object/class/text':  tf.VarLenFeature(tf.string),
      'image/object/class/label': tf.VarLenFeature(tf.int64),
      'image/object/difficult':   tf.VarLenFeature(tf.int64),
      'image/object/truncated':   tf.VarLenFeature(tf.int64),
      'image/object/view':        tf.VarLenFeature(tf.string),
    }
    def _parse_record(example):
        return tf.parse_single_example(example, feature_description)

    return dataset.map(_parse_record)


def show_record(filenames):
    """Show the TFRecord contents"""
    # Generate examples from TFRecord file/s.
    examples = parse_tfrecord(filenames)

    for example in examples:
        image_decoded = tf.image.decode_image(example['image/encoded'], channels=3).numpy()
        height = example['image/height'].numpy()
        width = example['image/width'].numpy()
        filename = bytes.decode(example['image/filename'].numpy())
        img_format = bytes.decode(example['image/format'].numpy())
        xmin_ = example['image/object/bbox/xmin']
        ymin_ = example['image/object/bbox/ymin']
        xmax_ = example['image/object/bbox/xmax']
        ymax_ = example['image/object/bbox/ymax']
        labels = example['image/object/class/label'] ## ID
        classes = example['image/object/class/text'] ## Name

        ## Extracted info
        print(filename, img_format, width, height)

        # Use OpenCV to preview the image.
        image = np.array(image_decoded, np.uint8)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
        # Draw the bboxes on image
        print(xmin_)
        for i in range(len(labels.values)):
            cls = bytes.decode(classes.values[i].numpy())
            label = labels.values[i]
            print("\tLabel: {}, Class: {}".format(label, cls))
            ymin = int(ymin_.values[i]*height)
            xmin = int(xmin_.values[i]*width)
            ymax = int(ymax_.values[i]*height)
            xmax = int(xmax_.values[i]*width)
            print(ymin,xmin,ymax,xmax)
            image = drawBoundingBox(image, [ymin,xmin, ymax,xmax], label)  ## t-l row,col, b-r row,col
            # Following exposes a bug in Eager execution, hence, not used.
            #image_decoded_ = tf.image.draw_bounding_boxes(image_decoded, [ymin,xmin, ymax, xmax], name=cls)

        # Show the result
        cv.imshow("image", image)
        if cv.waitKey() == 27:
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        type=str,
        default="train.record",
        help="The record file."
    )
    args = parser.parse_args()
    show_record(args.record)
    print("Done!")

