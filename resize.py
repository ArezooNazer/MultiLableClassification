# USAGE
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background

from shutil import copyfile
from imutils import paths
import numpy as np
import argparse
import cv2
import os



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="/home/arezoo/Desktop/keras-multi-label/dataset/glass-bottle",
	help="path to input dataset (i.e., directory of images)")

args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))

# initialize the data and labels
total = 0
i = 0
deletePath = "/home/arezoo/Desktop/keras-multi-label/delete/"
resizePath = "/home/arezoo/Desktop/keras-multi-label/resized/glass-bottle"
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    # try to load the image
    #print("[INFO] images path..." , imagePath)
    delete = False
    try:
        print("[INFO] reading images...")
        image = cv2.imread(imagePath)
        # if the image is `None` then we could not properly load it
        # from disk, so delete it
        if image is None:
            delete = True
    # if OpenCV cannot load the image then the image is likely
    # corrupt so we should delete it
    except:
        print("Except")
        delete = True
    #check to see if the image should be deleted
    if delete:
        print("**** deleting {}".format(imagePath))
        copyfile(imagePath , os.path.join(deletePath, "{}.jpg".format(str(total).zfill(8))) )
        total += 1
        os.remove(imagePath)
    else:  
        print(" [INFO] resizing {}".format(imagePath))
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        cv2.imwrite(os.path.join(resizePath , "{}.jpg".format(str(i).zfill(8))), image)
        i += 1
        