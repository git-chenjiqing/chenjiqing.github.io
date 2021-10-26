---
layout: post
title: How create a eye detector using deep learning?
subtitle: This applications determinates if your eyes are opened or closed
cover-img: /assets/img/demo_closed_img.jpg
thumbnail-img: /assets/img/demo_closed_img.jpg
share-img: /assets/img/path.jpg
tags: [deeplearning, computervision, python, code, tutorial, keras, tensorflow]
---

### Introduction

To teach something to the computer in some cases could be very difficult, image classification is a computer vision topic which become's very popular, in this topic we are interested to teach the machine to recognize objects present in a scene, here I brought an example of image classification problem, how to determine if a person is with opened or closed eyes?

In the classification task, commonly we use neural networks, to teach a neural network we need to have a dataset, in our case, our dataset is a set of images that will help us to conclude our objective, so we will show those images from the dataset to the neural network and it will return to us a model that can aswer our main question, "how to determine if a person is with opened or closed eyes".

### Dataset

The <b>dataset</b> that we are use in this task is the <b>CEW Dataset</b>, you could download it from [here](https://drive.google.com/uc?id=1niyedvpnATsWMnhcy_DfNNhPGc2J_G8V) it is composed by faces with closed and opened eyes.
Closed eyes | Opened eyes
:------------------------------:|:-------------------------:
![Closed Eyes](https://lh3.googleusercontent.com/kk_5Hj_uwptJa6WGNKeuJxw7-qbnn7aReMbi59iYWHwooeQQLDptdHePbPHulnNseTdyUQxgieObzvU0auZlJs-_PS3ZoGeH5iBnclqoUXIZjdAY1QL7klKasOlM6gb6AfbN_2MS=s100-p-k) | ![Opened Eyes](https://lh3.googleusercontent.com/iSbELyMtCInjeSTN_P4DDfHho6deaQtwNvMs_lp9FSosLzOIQNmzmA55yNV2sSymrcJV8T-8KAGbzV1bNe4CkdbMpowgkZIH7tRS2S8vIOFqJJ-2wUuADhYv9Yb5p0lyeyr_pQ1Z=s100-p-k)

### Preprosssing images

The first thing that we need to do when we are facing a classification problem is to understand what features really matters to solve the problem, for example, if we are interested in just eyes information, other stuff like nose, mouth,ears, or hair are useless information that we can discard because it does not make part of the task.

To remove the useless information from images we will do a preprocess steop using the lib <b>dlib</b> to detect landmarks from faces and crop the images on the interest region.

![Landmarks](https://lh3.googleusercontent.com/FeCdBHKsSeXTrT29-G3a18gsOB3hpS6NJHmTMdBAzcu2wC99vcPCJgkgOAVChr2Dk_-SznacEV8JgDzoLayZzPbMArMhoEFw0ty3pPredmOFwulzNRmI9BknsTHjY2jbbuATfFcB=w414-h315-p-k)

### Coding our preprocess program

Create a folder to our project and inside it create a file with any name you want, in my case i will use `preprocess.py`

If you already didn't download the CEW Dataset [here](https://drive.google.com/uc?id=1niyedvpnATsWMnhcy_DfNNhPGc2J_G8V)

Also download the landmark detector model of dlib [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), i will store it in an folder called models.

Lets get the user input from terminal to know the dataset path.

### Dependencies

- install dlib: `pip3 install dlib`
- install opencv: `pip3 install opencv-python`
- install imutils: `pip3 install imutils`

```python
import dlib
import cv2
import sys # to get the user input
from imutils import paths
if __name__ == "__main__":
    # get the user input
    if len(sys.argv)!=2:
        raise IOError(
                    """
                    usage: python3 preprocess.py dataset_path
                    """
                    )
    dataset_path = sys.argv[1]
    ###### load all images from dataset #####
    paths_list = list(paths.list_images(dataset_path))
    image_list = [] # struct that will store all images
    labels_list = [] # labels says if we have a closed or opened eyes
    image_name_list = [] # save this information to use later
    # store those informations one by one in this loop:
    for path in paths_list:
        image_list.append(cv2.imread(path))
        splitted_path = path.split()
        labels_list.append(splitted_path[-2])
        image_name_list.append(splitted_path[-1]):

```

### Training our net

After preprocess we need to train our network, to do it we will use the TensorFlow Keras API, which is a high-level API for build neural networks.

For training we will use a few techniques that will make our network "smarter", the techniques are:

- <b> Data augmentation </b>

  To make our network more precise we need to have lots of data, data augmentation is responsible to generate data using pre existing data, how it is possible? apply transformation such as rotations, flip and others in the image set, it also makes our dataset "more" pose invariant.

- <b>Transfer learning</b>

  We choose a pre-trained model to transfer its "abilities" to our network, which "ability" we are interested in? the object detection, so our network at first knows to identify objects but not eyes, for this reason, we will train for eyes.

- <b>Fine tuning</b>

  In the object classification the last layers of the neural network are responsible to return the answer about which object is present in the image, this task is also called prediction, to perform the prediction we also could train more layers than just the last layers, we could to train the hidden layers, this is known by fine-tuning.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D,MaxPool2D, Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import os
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def loadDataset(input_path):
    imagePaths = list(paths.list_images(input_path))
    data = []
    labels = []
    preprocess = lambda img: img_to_array(img)/255.0
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = load_img(imagePath, target_size=(64, 64))
        image = preprocess(image)
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    return (data,labels)

INIT_LR = 1e-4
EPOCHS = 40
BS = 32
DATASET_PATH = "./dataset/eyes/"
if __name__=="__main__":
    # Load all images and the labels
    #       OpenedFace e ClosedFace
    data,labels = loadDataset(DATASET_PATH)
    # Transform the labels into numbers
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    # "split" the dataset into 2 sets, one is the test set
    # and the other the train set, the test set have 20% of
    # the total images and the train have the rest
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42)
    # Generate data using data augmentation, apply
    # image transformations into the test and train set
    # to peform our network precision
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    # use the pretrained model mobilenetV2
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(64, 64, 3)))
    # Add some few aditional layers for predict
    headModel = baseModel.output
    # To reduce the number neurons of the dense layer
    headModel = MaxPool2D(pool_size=(2, 2))(headModel)
    # Transform the image matrix into a one dimenional array
    headModel = Flatten(name="flatten")(headModel)
    # Add some dense layer to calculate the weights
    # and improve the prediction
    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    # Also to add an dropout layer to avoid overfiting
    headModel = Dropout(0.5)(headModel)
    # Add 2 layers to make the prediction of closed or opened eyes
    headModel = Dense(2, activation="softmax")(headModel)
    # Join everything in a unique model
    model = Model(inputs=baseModel.input, outputs=headModel)
    # show the layers in the screen
    print(model.summary())
    # Fine tunning
    for layer in baseModel.layers[8:]:
    	layer.trainable = True
    # Apply the adam optimizer
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # We use the binary_crossentropy because is a good
    # choise due we only have 2 options (open or closed eyes)
    model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])
    print("Train started")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    # evaluate our test
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)
    #
    # print the results in the screen
    #
    print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=lb.classes_))
    print("saving the model")
    # save the model into the disk
    model.save("models/eyes_detector_model.h5", save_format="h5")
    # Plot the result
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot( H.history["loss"], label="train_loss")
    print(H.history.keys())
    for key in H.history.keys():
        print(f'key:{key} {H.history[key]}')
    plt.plot( H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot_novo")

```

### Dependencies

- Download the face detection model [here](https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel)

- Download the face detection proto txt file [here](https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/deploy.prototxt.txt)

### Using our model

```python
import cv2
from cv2 import dnn
import numpy as np
import time
import dlib
from imutils import face_utils
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from src.face_detector import FaceDetector
from src.face_landmark_predictor import FaceLandmarkPredictor
import time
def getBoudingboxFromLandmarksList(landmaks_list):
    bouding_box = [landmaks_list[0],landmaks_list[-1]]
    for landmark in landmaks_list:
        if bouding_box[0][0] < bouding_box[0][0]:
            bouding_box[0][0] = landmark[0]
        if landmark[1] < bouding_box[0][1]:
            bouding_box[0][1] = landmark[1]
        if landmark[0] > bouding_box[1][0]:
            bouding_box[1][0] = landmark[0]
        if landmark[1] > bouding_box[1][1]:
            bouding_box[1][1] = landmark[1]
    return bouding_box
if __name__ == "__main__":
    # open the camera
    cap = cv2.VideoCapture(0)
    # load the face detection model
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    face_landmark_predictor = FaceLandmarkPredictor()
    range_idxs = []
    range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"])
    range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"])
    range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["right_eyebrow"])
    range_idxs.append(face_utils.FACIAL_LANDMARKS_68_IDXS["left_eyebrow"])
    # load our model
    model = load_model("./models/eyes_closed_open_model_64_64.h5")
    while cap.isOpened:
        # get image from camera
        ret, img = cap.read()
        # resize the image for match with the network
        # face detection expected size
        img = cv2.resize(img,(300,300))
        # copy an image to be our displayed image
        show_img = img.copy()
        # convert the image to gray to be the dlib input image
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # detect faces on the image
        for box in face_detector.detectFaces(img):
            # detect landmarks at the faces
            landmark_list = face_landmark_predictor.predictLandmarks(img_gray,box)

            # get the Region of interest of the face
            isInRange = lambda idx, idx_range :  True if idx>=idx_range[0] and idx<=idx_range[1] else False
            good_landmarks_list = []
            cv2.rectangle(show_img,(box[0],box[1]),(box[2],box[3]),(255,0,0))
            for i in range(48):
                for range_idx in range_idxs:
                    if isInRange(i, range_idx) or i == 30:
                        good_landmarks_list.append(list(landmark_list[i]))
                        #cv2.circle(show_img,landmark_list[i],1,(255,0,0))
            bb = getBoudingboxFromLandmarksList(good_landmarks_list)

            pt1 = (bb[0][0],bb[0][1])
            pt2 = (bb[1][0],bb[1][1])
            eye_img = None
            # preprocess the image to be  our neural network

            try:
                # crop image
                eye_img = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
                # resize image
                eye_img = cv2.resize(eye_img,(64,64),interpolation=cv2.INTER_AREA)
            except:
                continue
            # conver the image to values between 0 and 1
            eye_img = eye_img/255
            # convert to one dimentional array
            eye_img = img_to_array(eye_img)
            eye_img = np.expand_dims(eye_img, axis=0)
            # predict
            (closed, opened) = model.predict(eye_img)[0]

            label = f'Closed: {round(closed*100,2)}%'
            score = closed
            color = (0,0,255)

            # show the information to the user

            if opened > closed:
                label = f'Opened: {round(opened*100,2)}%'
                score = opened
                color = (0,255,0)
            if score>0.9:
                cv2.putText(show_img,label,(int(box[0]),box[3]+10),cv2.FONT_HERSHEY_PLAIN,1,color,2)
                print(label)
                cv2.rectangle(show_img,(pt1),(pt2),color)

        cv2.imshow("img",show_img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

```
