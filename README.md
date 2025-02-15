# Convolutional Neural Networks Deep Dive

## Overview
This project is a deep dive into **Convolutional Neural Networks (CNNs)**, a type of deep learning model specialized for grid-like data structures such as images. The notebook explores various CNN architectures, their key components, and implements a model using the CIFAR-100 dataset.

## Table of Contents
1. [Introduction to CNNs](#introduction-to-cnns)
2. [CNN Architectures](#cnn-architectures)
3. [Key Components of CNNs](#key-components-of-cnns)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Implementation](#model-implementation)
6. [Performance Evaluation](#performance-evaluation)
7. [Data Augmentation](#data-augmentation)
8. [Results and Analysis](#results-and-analysis)
9. [Installation and Usage](#installation-and-usage)
10. [References](#references)

## Introduction to CNNs
CNNs are designed to process structured grid data, leveraging **convolutional layers** to extract hierarchical features automatically. This reduces the need for manual feature engineering and enhances model performance for tasks such as image recognition and classification.

## CNN Architectures
This project explores various CNN architectures, including:
- **LeNet-5** (1998): Early CNN for handwritten digit recognition.
- **AlexNet** (2012): Introduced deep networks with ReLU and dropout.
- **VGGNet** (2014): Used deep convolutional layers with small filters (3x3).
- **GoogLeNet/Inception** (2014): Introduced multi-scale feature extraction.
- **ResNet** (2015): Used residual connections for ultra-deep networks.
- **YOLO (You Only Look Once)** (2015): Real-time object detection.
- **U-Net** (2015): Medical image segmentation.
- **DenseNet** (2017): Improved feature reuse across layers.

## Key Components of CNNs
- **Convolutional Layers**: Extract spatial features from images using filters.
- **Activation Functions**: Introduce non-linearity (ReLU, Sigmoid, Softmax).
- **Pooling Layers**: Reduce spatial dimensions to enhance computational efficiency.
- **Fully Connected Layers**: Flatten feature maps and make predictions.
- **Dropout**: Prevent overfitting by randomly dropping neurons during training.

## Dataset Preparation
- **Dataset**: CIFAR-100 dataset from the University of Toronto.
- **Preprocessing Steps**:
  - Data extraction and formatting.
  - Normalization of pixel values.
  - Splitting into training and validation sets.

## Model Implementation
- Implemented a **custom CNN architecture** using TensorFlow/Keras.
- Model contains **three convolutional layers**, **max pooling layers**, **dropout**, and **fully connected layers**.
- **Optimization**: Adam optimizer, Sparse Categorical Cross-Entropy loss function.
- **Training Setup**: 30 epochs with early stopping and model checkpointing.

## Performance Evaluation
- Accuracy and loss graphs for training and validation.
- Model hyperparameter tuning (L2 regularization, dropout rate, learning rate adjustments).
- Best model performance recorded at **epoch 30**.

## Data Augmentation
- Implemented **random flipping** and **random rotation** to improve model generalization.
- Retrained model using augmented dataset.
- Performance comparison between the original and augmented models.

## Results and Analysis
- Improved validation accuracy after applying data augmentation.
- Comparison of different CNN architectures in terms of **accuracy and computational efficiency**.
- Next steps involve **testing pre-trained architectures (VGG16, ResNet)** to further enhance performance.

## Installation and Usage
### Requirements
- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib / Seaborn

### Running the Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CNN_Deep_Dive.git
   cd CNN_Deep_Dive
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook Convolutional_Neural_Networks_Deep_Dive.ipynb
   ```

## References
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [Convolutional Neural Networks - Medium](https://medium.com/analytics-vidhya/convolution-operations-in-cnn-deep-learning-compter-vision-128906ece7d3)
- TensorFlow Documentation: https://www.tensorflow.org/

---
### Author
*Your Name / GitHub Username*
