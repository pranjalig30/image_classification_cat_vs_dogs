
# <center> Cat vs. Dogs Prediction

<center> <img src="https://github.com/brandon-park/Transfer_learning_cat_vs_dog/blob/main/Header_Cats-vs-Dogs-951x512.png?raw=true" width="70%"/>


## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Transfer Learning](#transfer-learning)
- [Code Summary](#code-summary)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Introduction
This repository contains a Jupyter notebook that implements a Convolutional Neural Network (CNN) to classify images of cats and dogs using both a custom CNN and a pre-trained VGG16 model. This notebook is designed to be run on Kaggle.

**Kaggle Competition**: [dogs-vs-cats-redux-kernels-edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

## Dataset Description
- **Training Set**: 25,000 labeled images of cats and dogs.
- **Test Set**: 12,500 images requiring probability predictions for being a dog (1 = dog, 0 = cat).

Due to Kaggle's limited free GPU/RAM, image sizes are reduced to ensure the training process completes within the available computing power.

## Transfer Learning
### Benefits
- Reduces the need for extensive labeled data for new models.
- Enhances efficiency in developing and deploying machine learning models.
- Provides a generalized approach to solving related problems.
- Allows models to be trained in simulations rather than real-world environments.

## Code Summary
### Data Loading and Preprocessing
- Import libraries: numpy, pandas, TensorFlow.
- Load and preprocess the dataset.

### Model Building
- Custom CNN model using Keras' Sequential API.
- Pre-trained VGG16 model for transfer learning.

### Model Compilation and Training
- Compile with Adam optimizer and binary crossentropy loss.
- Train the model and evaluate performance.

### Model Evaluation and Predictions
- Evaluate on the validation dataset.
- Predict on the test dataset and process results.

## Evaluation
Submissions are scored on the log loss:

\[ \text{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] \]

where:

- \( n \) is the number of images in the test set
- \( \hat{y}_i \) is the predicted probability of the image being a dog
- \( y_i \) is 1 if the image is a dog, 0 if cat
- \( \log() \) is the natural (base e) logarithm

A smaller log loss is better.


## Conclusion
This notebook demonstrates effective image classification using custom and pre-trained CNN models. The pre-trained VGG16 model achieved a high validation accuracy of 98.04%, showcasing the power of transfer learning.

## Future Work
- Further tuning of hyperparameters and model architecture.
- Experimentation with other pre-trained models.
- Application of additional data augmentation techniques to enhance model robustness.

Feel free to explore the notebook for detailed code and outputs.
