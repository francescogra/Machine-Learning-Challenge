# Machine-Learning-Challenge

# IMAGE-BASED LOCALISATION

The examination challenge in question was assigned to the students of the Machine Learning course held by Prof. Giovanni Maria Farinella within the Master of Science in Computer Science course at the University of Catania.

The dataset used in this challenge is a reduced version of the dataset used in the following scientific paper (http://iplab.dmi.unict.it/EgocentricShoppingCartLocalization/): 

The challenge is to tackle an image-based localisation problem, i.e. to construct an algorithm that, given an image taken in a known space, allows us to infer the position from which the image was taken (see image below). In particular, we will deal with indoor localisation. Furthermore, we will consider the 2D classification problem, instead of the more general 3D localisation problem. 

The challenge proposes two development methods for the previously described localisation, classification or regression, in this documentation we will deal with classification-based development. 

# Classification-based localisation: 
each image has a label indicating in which part of the building the image was acquired. To obtain the labels, the building is generally divided into non-overlapping zones, so that each image belongs to one and only one class;

![alt text](https://github.com/francescogra/Machine-Learning-Challenge/blob/main/slide1.png "Machine Learning Challenge")

# Evaluation
The algorithm is to be evaluated by reporting:
- Clasification: Accuracy, confusion matrix, F1 scores relative to the different classes and mF1 scores;

# Dataset 
The dataset consists of 19531 images acquired inside a supermarket, divided into Each image is labelled with respect to: - Class to which it belongs. The supermarket was divided into 16 macro-areas and each image was assigned to one of these macro-areas. The classes are numbers ranging from 0 to 15. The figure below shows the plot of the positions of the images in the training set. The colours indicate the classes to which each image belongs. The labels of the training and validation set are provided, while those of the testing set are not made public.

![alt text](https://github.com/francescogra/Machine-Learning-Challenge/blob/main/slide2.png "Machine Learning Challenge")
