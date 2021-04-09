---
layout: post
title: How create a eye detector using deep learning?
subtitle: This applications determinates if your eyes are opened or closed
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [deeplearning, computervision,python,code,tutorial,keras,tensorflow]
---

### Introduction

To teach something to the computer in some cases could be very difficult, image classification is a computer vision topic that's become very popular, in this task we are interested to teach the machine to recognize objects present in a scene, here I brought an example of image classification problem, how to determine if a person is with opened or closed eyes?

In the classification task, commonly we use neural networks, to teach a neural network we need to have a dataset,  in our case, our dataset is a set of images that will help us to conclude our objective, so we will show those images from the dataset to the neural network and it will return to us a model that can recognize eyes.

### Dataset


The <b>dataset</b> that we are use in this task is the <b>CEW Dataset</b>, you could download it from here [here](https://drive.google.com/uc?id=1niyedvpnATsWMnhcy_DfNNhPGc2J_G8V) this dataset is composed by faces with closed and opened eyes.
Closed  eyes    |  Opened eyes
:------------------------------:|:-------------------------:
![Closed Eyes](https://lh3.googleusercontent.com/kk_5Hj_uwptJa6WGNKeuJxw7-qbnn7aReMbi59iYWHwooeQQLDptdHePbPHulnNseTdyUQxgieObzvU0auZlJs-_PS3ZoGeH5iBnclqoUXIZjdAY1QL7klKasOlM6gb6AfbN_2MS=s100-p-k)     |       ![Opened Eyes](https://lh3.googleusercontent.com/iSbELyMtCInjeSTN_P4DDfHho6deaQtwNvMs_lp9FSosLzOIQNmzmA55yNV2sSymrcJV8T-8KAGbzV1bNe4CkdbMpowgkZIH7tRS2S8vIOFqJJ-2wUuADhYv9Yb5p0lyeyr_pQ1Z=s100-p-k)

### Preprosssing images

The first thing that we need to do when we are facing a classification problem is to understand what features really matters for your problem, for example, if we are interested in just eyes information, other stuff like nose, mouth,ears, or hair are useless information that we can discard because it does not make part of the task.

To remove this useless information from images we will use the lib <b>dlib</b> to detect landmarks from faces and crop the images on the interest region.

![Landmarks](https://lh3.googleusercontent.com/FeCdBHKsSeXTrT29-G3a18gsOB3hpS6NJHmTMdBAzcu2wC99vcPCJgkgOAVChr2Dk_-SznacEV8JgDzoLayZzPbMArMhoEFw0ty3pPredmOFwulzNRmI9BknsTHjY2jbbuATfFcB=w414-h315-p-k)

* install dlib:  `pip3 install dlib`
* install opencv: `pip3 install opencv-python`
* install imutils: `pip3 install imutils`
### Coding our preprocess program

Create a folder to our project and inside it create a file with any name you want, in my case i will use `preprocess.py`


If you already didn't download the CEW Dataset  [here](https://drive.google.com/uc?id=1niyedvpnATsWMnhcy_DfNNhPGc2J_G8V)

Also download the landmark detector model of dlib [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), i will store it in an folder called models.


Lets get the user input from terminal to know the dataset path.


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