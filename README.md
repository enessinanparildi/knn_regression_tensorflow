# TensorFlow Regression with K-Nearest Neighbors and Soft Assignments
Tensorflow implementation of KNN regression with hard and soft responsibility. For hard responsiblity, change kset at run_main to define which values of k hyperparameter should be searched. For soft responsibility, to search hyperparamaters, modify lambdaset at run_main
This project implements a regression model using TensorFlow, incorporating both hard k-nearest neighbors and soft assignment approaches.

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Key Functions](#key-functions)
4. [Usage](#usage)
5. [Model Details](#model-details)

## Overview

This code demonstrates a regression task using TensorFlow. It implements both hard k-nearest neighbors and soft assignment methods for prediction. The model is trained, validated, and tested on synthetic data.

## Dependencies

- TensorFlow
- NumPy
- Matplotlib

## Key Functions

### Data Generation

- `make_dataset()`: Generates synthetic data for training, validation, and testing.

### Distance Calculation

- `euclid_distance(X, Z)`: Calculates the Euclidean distance between two sets of points.

### Responsibility Assignment

- `hardresponsibility(D, k, graph)`: Assigns hard responsibilities based on k-nearest neighbors.
- `softresponsibility(D, lambd, graph)`: Assigns soft responsibilities using a lambda parameter.

### Loss Calculation

- `mseloss(prediction, target)`: Calculates mean squared error loss.

### Main Operations

- `main_operation(i, distancesettrain, kvalue, lambdaval, n_train, training_target, graph, hard_mode)`: Performs the main prediction operation.
- `hard_op(distances, kvalue, graph, training_target, n_train)`: Performs hard assignment operation.
- `soft_op(distances, lvalue, graph, training_target, n_train)`: Performs soft assignment operation.

### Model Training and Evaluation

- `main_graph(traindata, validdata, testdata, kset, lambdaset)`: Builds and runs the TensorFlow graph for training, validation, and testing.

### Visualization

- `plot_results(testdata, testpred, chosenk)`: Plots the test results.

## Usage

1. Import the necessary libraries:
   ```python
   import tensorflow as tf
   import numpy as np
   import matplotlib.pyplot as py
