
# <center> Cats Vs Dogs Image Classification using Transfer Learning

<center> <img src="https://github.com/brandon-park/Transfer_learning_cat_vs_dog/blob/main/Header_Cats-vs-Dogs-951x512.png?raw=true" width="70%"/>

## TOC:

1. [Introduction](#Introduction)
2. [Transfer Learning](#Transfer-Learning)
3. [Code Summary](#Code-Summary)

## Introduction <a name="Introduction"></a>
This repository contains a Jupyter notebook that implements a Convolutional Neural Network (CNN) to classify images of cats and dogs. The notebook explores two different model architectures: a custom CNN and a pre-trained VGG16 model.

This Notebook should be run in Kaggle to import the dataset.

- Competition name: dogs-vs-cats-redux-kernels-edition
 
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

#### Dataset Description
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).

Since the Kaggle's free GPU/RAM is limited, we need to decrease the size of the raw image file so that entire traning process can be finished within the computing power.

## Transfer Learning <a name="Transfer-Learning"></a>

### Benefits of Transfer Learning

In transfer learning, a machine learning model is trained on one kind of problem, and then used on a different but related problem, drawing on the knowledge it already has while learning its new task. This could be as simple as training a model to recognize giraffes in images, and then making use of this pre-existing expertise to teach the same model to recognize pictures of sheep.

The main benefits of transfer learning for machine learning include:

- Removing the need for a large set of labelled training data for every new model.
- Improving the efficiency of machine learning development and deployment for multiple models.
- A more generalised approach to machine problem solving, leveraging different algorithms to solve new challenges.
- Models can be trained within simulations instead of real-world environments.


## Code Summary
The notebook includes the following key sections:

1. **Data Loading and Preprocessing:**
   - Import necessary libraries such as numpy, pandas, and TensorFlow.
   - Load and preprocess the dataset.

2. **Model Building:**
   - Build the custom CNN model using Keras' Sequential API.
   - Implement the pre-trained VGG16 model for transfer learning.

3. **Model Compilation and Training:**
   - Compile the model with the Adam optimizer and binary crossentropy loss.
   - Train the model on the dataset and evaluate its performance.

4. **Model Evaluation and Predictions:**
   - Evaluate the model on the validation dataset.
   - Make predictions on the test dataset and process the results.

#### Evaluation
Submissions are scored on the log loss:

LogLoss=−1n∑i=1n[yilog(y^i)+(1−yi)log(1−y^i)],
where

n is the number of images in the test set
\\( \hat{y}_i \\) is the predicted probability of the image being a dog
\\( y_i \\) is 1 if the image is a dog, 0 if cat
\\( log() \\) is the natural (base e) logarithm
A smaller log loss is better.

### Conclusion
This notebook provides a comprehensive implementation of image classification using both custom and pre-trained CNN models. The pre-trained VGG16 model achieved high validation accuracy, demonstrating the effectiveness of transfer learning for this task.

### Future Work
- Further tuning of hyperparameters and model architecture.
- Experimentation with other pre-trained models.
- Application of additional data augmentation techniques to improve model robustness.

Feel free to explore the notebook for detailed code and outputs.
