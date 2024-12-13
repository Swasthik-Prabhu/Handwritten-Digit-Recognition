# Handwritten Digit Recognition using MNIST Dataset

This project implements a machine learning model to recognize handwritten digits from the MNIST dataset using TensorFlow and Keras. The goal is to build a simple Convolutional Neural Network (CNN) model for digit classification, which can classify digits from 0 to 9.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Project Overview

The MNIST dataset consists of 60,000 28x28 grayscale images of handwritten digits (0-9) for training and 10,000 images for testing. The task is to train a model to recognize the digits from these images. We will use a Convolutional Neural Network (CNN) to achieve this.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition


2.Install the required dependencies:
  ```
  install -r requirements.txt

3.To train the model on the MNIST dataset, run the following script:
   ```
   python train_model.py


Load and preprocess the MNIST dataset
Build and train a CNN model
Evaluate the model on the test dataset
Save the trained model to a file mnist_model.h5


4.To make predictions on new images, use the script predict.py:
   ```
    python train_model.py
   ```

## Model Architecture
The model uses a simple Convolutional Neural Network (CNN) with the following layers:

Conv2D layer with 32 filters and a 3x3 kernel
MaxPooling2D layer to downsample the image
Conv2D layer with 64 filters and a 3x3 kernel
MaxPooling2D layer to further downsample the image
Flatten layer to convert the 2D matrix to 1D
Dense layer with 128 units and ReLU activation
Dense layer with 10 units (corresponding to the 10 digit classes) and Softmax activation

## Training
The model is trained using the Adam optimizer and sparse categorical crossentropy loss.
The model is trained for 10 epochs with a batch size of 32.
You can increase the number of epochs for better performance.

## Evaluation
After training, the model is evaluated on the test dataset:


## License
This project is licensed under the MIT License - see the LICENSE file for details.

   ---
   
### Steps to Upload to GitHub:
1. Create a new repository on GitHub.
2. Clone the repository to your local machine:
      ```
      https://github.com/Swasthik-Prabhu/Handwritten-Digit-Recognition.git

3.Copy your project files (including the README.md) into the cloned repository folder.
4.Commit and push the changes:
   ```
   git add .
   git commit -m "Initial commit"
   git push origin main

   
   

