# Handwritten Equation Solver
A deep learning approach to developing a Handwritten Equation Solver

## Objective
In this project we develop a system to automatically solve handwritten equation solvers from their images.
We demonstrate our solution by constructing a system for recognizing and producing solutions of 2 and 3 variable simultaneous linear equation, quadratic, and cubic equations just by feeding their images to the model.

## Overview

In this paper we demonstrate an image based equation solver using a webcam installed on the computer. All computations are done locally using Python, without the need to relay data to a server.  

1. **Data Pre-Processing**: We begin by reading in each image and pre-processing it in order to reduce the variations between different examples. This allows the system to create a much clearer picture of the distinctions between different labels.  

2. **Training**: Different machine learning models were trained and tested on the pre-processed images. The model with the highest accuracy/lowest loss was finally selected.    

3. **Solving the equation**: The captured image(from the webcam) was fed into the model and predicted coefficients of the variables (in the equation) are then given to a predefined function which gives the solutions of the equation.We have also developed a graphic user interface to allow the user to specify the kind of equation for which they want solution.   

## Dataset

The DATA from [CHROME](https://www.isical.ac.in/~crohme/CROHME_data.html)
There are 75 different mathematical symbols present in the dataset. Out of these 75 classes we require only 18 classes for our present machine learning model. The pixel arrays of the images in the dataset are normalized for the size of 45x45.     
The dataset consists of 165,000  images of size 45 x 45 from 18 different classes. A train to test split of 0.2 was performed for training the model.

## Detailed Explanation

1. Image Pre-Processing and Detection: The objective of pre-processing is to take the captured image and segment the characters using the OpenCv library and python.

* The image is converted to grayscale and blurred to reduce lines on lined paper as well as noise.
* Further reduction of noise is done using the local denoising technique via a predefined function of OpenCv.
* The image is then binarized using adaptive thresholding . This binarized image is then inverted and segmented using the findContours OpenCV function.
* After the characters are segmented, they need to be normalized to the dataset of characters. The objective of character normalization is to make any character match as much as possible to the dataset.
* The dataset characters have a size of 45x45 pixels. To normalize, the characters binarized and zero padded and then finally scaled to the scale of 45x45 pixels.

2. **Network Structure and Training**: The structure of the model was inspired from AlexNet and consists of two convolutional layers, 2 max pooling layers, 3 dropout layers, and 4 fully connected layers with a total of 6.7 M trainable parameters.  Adam Optimizer with cross entropy loss function was used to train the model in 50 iterations with a test accuracy of 95.72% and test loss of 0.3325.

3. 

