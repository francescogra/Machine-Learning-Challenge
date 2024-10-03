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

# METHOD
The neural network model used to solve our image classification problem is the Resnet 50, built with the help of PyTorch


# Technical details of the solution 
Datasets: contains the train and test models 
datasets: contains the trained models/checkpoints 
train.py : To train the data 
predict.py : to predict from the trained model or to separate the images into separate folders
clearDataset.py: script used to clean up the train dataset with the images in the test set, in order to make predictions about images the model has never seen
class_mapping.json: here are the classes in json format 
hyper_params.json: the json file contains all hyper patameters and their notations. 

The data set has 16 classes, i.e. the 16 macro-areas of the supermarket, from 0 to 15. So here we intend to form a model that should be able to distinguish all these departments. The dataset has been arranged into train and test datasets Each of these folders contains the total number of images provided by the dataset described above. The train folder consists of 80% of the images while the test folder consists of the remaining 20%.
Each of them has been divided into 16 folders,
I have organised the dataset in this way, so that in each folder I have all the images from that particular department of the supermarket.


# Conclusion:

![alt text](https://github.com/francescogra/Machine-Learning-Challenge/blob/main/slide3.png "Machine Learning Challenge")

This challenge allowed me to personally apply the theory studied previously, going step by step to construct an algorithm capable of predicting by classification through a photo its department of origin.
Starting from the basic logic, I learnt that solving such a problem involves steps, which can be listed as follows
- Data acquisition
- Organisation of the dataset
- Normalising the dataset
- Fitting the data to the model
- Training the model
- Logical inference with trained model
- Review of some metrics needed to understand and qualify the work of the model such as: Pixels matrix, Features matrix, Raw pixel accuracy, Histogram accuracy.
In this paper, we examined a code that uses the ResNet model for image classification. We have seen that the ResNet model is designed to solve the gradient problem when training deep neural networks by using skip connection branches to maintain the error gradient even in deep neural networks. We also saw that the ResNet model consists of a basic part and a class part.

The base part consists of a sequence of convolution and normalisation layers, while the class part consists of a series of fully-connected layers followed by an output layer that produces the probabilities for the different classes.
During the training of the model, the backpropagation algorithm was used to calculate the errors made by the model and update the network weights in order to reduce errors during the subsequent prediction. The optimiser used was SGD (Stochastic Gradient Descent) with a learning rate scheduler to dynamically change the learning rate as the training progressed.
I realised that neural network models use a structure based on a network of artificial neurons that have been trained to recognise patterns in input features.

During the validation of the model, a loss value of 0.1640 and an accuracy of 95.31% was obtained. This means that the model committed an average error of 0.1640 during prediction and correctly guessed the class of approximately 95.31% of the examples in the validation dataset.
Overall, the performance of the ResNet model seems to be good, especially considering that it is a deep neural network. However, it is important to consider that the performance of the model may vary depending on several factors, such as the complexity of the model, the number of epochs used for training, the size of the dataset, etc.
Neural network models tend to perform better than other models for most applications, especially for large and highly complex data. However, neural network models may take longer to train than, for example, a KNN-type model.
