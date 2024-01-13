# Handwritten Equation Solver
A deep learning approach to developing a Handwritten Equation Solver

## Objective
In this project we develop a system to automatically solve handwritten equation solvers from their images.
We demonstrate our solution by constructing a system for recognizing and producing solutions of 2 and 3 variable simultaneous linear equation, quadratic, and cubic equations just by feeding their images to the model.

## Overview

In this paper we demonstrate an image based equation solver using a webcam installed on the computer. All computations are done locally using Python, without the need to relay data to a server.  

We begin by reading in each image and pre-processing it (as outlined in the later part of the paper) in order to reduce the variations between different examples. This allows the system to create a much clearer picture of the distinctions between different labels.  

In the next step, we train and test several different machine learning models on our pre-processed images. We finally chose the machine learning algorithm which gives the highest accuracy and the lowest loss.    

Finally, we feed the captured image(from the webcam) to the  machine learning model in order to get the predictions. The predicted coefficients of the variables (in the equation) are then given to predefined function in python which gives the solutions of the equation.We have also developed a graphic user interface to allow the user to specify the kind of equation for which they want solution.   
