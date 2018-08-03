# USAGE
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split
from pyimagesearch.smallervggnet import VGGNet
import matplotlib.pyplot as plt
from shutil import copyfile
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

path = "/media/sf_shared/keras-multi-label/"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default=path + "dataset/",
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", default=path+"trash2.model",
	help="path to output model")
ap.add_argument("-l", "--labelbin", default=path + "tmlb2.pickle",
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default=path + "tplot2.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 2
INIT_LR = 1e-3
BS = 10
IMAGE_DIMS = (224,224, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []
total = 0
deletePath = "/home/arezoo/Desktop/keras-multi-label/delete/"
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    # try to load the image
    #print("[INFO] images path..." , imagePath)
    delete = False
    try:
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
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        # extract set of class labels from the image path and update the
        # labels list
        l = label = imagePath.split(os.path.sep)[-2].split("-")
        labels.append(l)
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=0, width_shift_range=0,
	height_shift_range=0, shear_range=0, zoom_range=0,
	horizontal_flip=False, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = VGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])