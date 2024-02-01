# Satellite Image Classification with TensorFlow

This repository contains the code and documentation for a satellite image classification project using TensorFlow. The goal of the project is to accurately predict the category of satellite images by building a convolutional neural network (CNN).

## Project Overview

The model is designed to classify satellite images into four categories: cloudy, desert, green_area, and water. The dataset consists of labeled images collected from various sensors and Google Maps snapshots.

## Key Features

- Data preparation using TensorFlow's ImageDataGenerator for real-time data augmentation.
- Understanding the functionality of `flow_from_directory` for efficient data input pipeline.
- Visualization of the model's predictions on satellite images.
- Implementation of a CNN with several convolutional layers, max-pooling, dropout, and dense layers.
- Usage of early stopping to prevent overfitting.
- Evaluation of the model's performance using accuracy, precision, recall, and F1-score metrics.

## Dataset

The dataset used for training and validation is the RSI-CB256 dataset, which includes satellite images across four classes with different environmental features.

### 4 class Images:
![1](https://github.com/Hasanshovon/Satellite-image-classification-A-TensorFlow-approach/assets/26182608/08d5f5ad-cdb8-46f3-bf03-06648b72d80a)
![2](https://github.com/Hasanshovon/Satellite-image-classification-A-TensorFlow-approach/assets/26182608/bb91b10b-627e-4b09-85c7-9f16f978841a)

![3](https://github.com/Hasanshovon/Satellite-image-classification-A-TensorFlow-approach/assets/26182608/6aa86547-620e-4301-a581-64caeba3d7cd)
![4](https://github.com/Hasanshovon/Satellite-image-classification-A-TensorFlow-approach/assets/26182608/df4a7242-9c25-4db7-a334-db21a824e1a9)


## Model Architecture

The model is a CNN with the following layers:

- Conv2D with ReLU activation, followed by BatchNormalization and MaxPooling.
- Dropout layers to reduce overfitting.
- A Flatten layer followed by a Dense layer with ReLU activation.
- The output layer is a Dense layer with a softmax activation function to output class probabilities.

## Results

The trained model achieved an accuracy of approximately 94.23% on the test set. The detailed classification report is available in the repository, showing precision, recall, and F1-score for each class.

## Requirements

The project requires the following libraries:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- Pandas
