> 🇺🇸 This README is in English.  
> 🇧🇷 [Clique aqui para a versão em português.](README.md)

# 🧠 Deep Dive into Convolutional Neural Networks (CNNs)

## 🔍 Overview
This project offers an in-depth exploration of the core concepts, layers, and operations involved in **Convolutional Neural Networks (CNNs)**. Using the CIFAR-100 image dataset, custom neural networks are built from scratch with a focus on both didactic and practical purposes.

It’s ideal for demonstrating expertise in computer vision, low-level neural architecture design, and training best practices.

---

## 🎯 Objectives
- Deeply understand the **fundamental layers of CNNs** (conv2d, pooling, batch norm, dropout, etc.)
- Implement a **custom architecture** using Keras
- Apply **data augmentation** and regularization techniques
- Evaluate the model using metrics, learning curves, and confusion matrix

---

## 🧠 Dataset
- 📚 **Source:** [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- 🔢 **Structure:** 60,000 32x32 images (50,000 train + 10,000 test)
- 🔍 **Task:** Classification of 100 object categories

---

## 🏗️ Model Architecture
The CNN was built **from scratch** and includes:
- `Conv2D` layers with varying filter sizes and strides
- `MaxPooling2D` and `Dropout` for overfitting control
- `BatchNormalization` for stabilization
- Fully connected (Dense) layers for final decision
- `ReLU` activations + `Softmax` output layer

---

## 🧪 Training & Evaluation
- **Loss function:** `SparseCategoricalCrossentropy`
- **Optimizer:** `Adam`
- **Epochs:** 20
- **Additional techniques:** Data Augmentation, EarlyStopping

### 🔍 Evaluation Metrics
- Accuracy (training and validation)
- Learning curves
- Confusion matrix
- Per-class accuracy (top-1)

---

## 📦 Tools & Libraries
- TensorFlow / Keras
- Matplotlib / Seaborn
- NumPy / Pandas
- Google Colab (development environment)

---

## 🔁 Future Improvements
- Apply **Transfer Learning** with networks like ResNet, EfficientNet
- Add **Grad-CAM** visualizations
- Export the model for production use (API or mobile deployment)
- Train on other datasets (e.g., FashionMNIST, Tiny ImageNet)

---

### 👨‍💻 About the Author

**André Rizzo**  
📊 Senior Data Scientist | Statistician | MBA in AI and Big Data (USP)  
🧠 Specialist in Deep Learning, Computer Vision and Predictive Modeling  
📍 Rio de Janeiro, Brazil  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/andrerizzo1)  
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?logo=github&logoColor=white)](https://github.com/andrerizzo)  
[![Email](https://img.shields.io/badge/Email-andrerizzo@hotmail.com-D14836?logo=gmail&logoColor=white)](mailto:andrerizzo@hotmail.com)

---

*Last updated: March 30, 2025*
