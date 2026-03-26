# CIFAR-10 Image Classification: Custom CNN vs. Transfer Learning (ResNet18)

## Project Overview

This project is part of a Deep Learning assessment focusing on image classification using the **CIFAR-10** dataset. The primary objective is to build, train, and compare two different Deep Learning approaches:
1. A Convolutional Neural Network (CNN) built entirely from scratch.
2. A Transfer Learning approach utilizing a pre-trained model (ResNet18).

The project evaluates both models based on accuracy, precision, recall, and F1-score, and discusses the trade-offs between building a custom architecture versus leveraging pre-trained weights.

## Dataset

The dataset used is **CIFAR-10**, which consists of 60,000 32x32 color images across 10 mutually exclusive classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). 
* **Training set:** 50,000 images
* **Testing set:** 10,000 images

The dataset is loaded and managed using PyTorch's `torchvision.datasets` module.

## Data Preprocessing

Data preprocessing and augmentation techniques were applied using `torchvision.transforms`:

* **For the Custom CNN:**
  * Converted images to PyTorch Tensors.
  * Normalized color channels with a mean and standard deviation of `(0.5, 0.5, 0.5)`.
  * Applied `RandomHorizontalFlip()` to the training data to prevent overfitting.
* **For Transfer Learning (ResNet18):**
  * **Resizing:** Because ResNet18 expects 224x224 inputs, the 32x32 CIFAR-10 images were upscaled using `Resize(224)`.
  * **ImageNet Normalization:** Used the specific mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]` required by models pre-trained on ImageNet.

## Model Architectures

### 1. Custom CNN
A straightforward convolutional architecture designed to be lightweight and fast to train on 32x32 images:
* **3 Convolutional Layers** (32, 64, and 128 filters) with ReLU activation and 2x2 Max Pooling.
* **2 Fully Connected Layers** (Flattened to 2048 -> 512 -> 10 output classes).
* **Dropout (p=0.25)** applied before the final layer to improve generalization.

### 2. Transfer Learning (ResNet18)
Utilized the pre-trained `ResNet18` model. Two different strategies were tested:
* **Feature Extraction:** Froze 100% of the pre-trained layers and replaced only the final fully connected (`fc`) layer to output 10 classes instead of 1000.
* **Fine-Tuning:** Froze the early layers but *unfroze* the final convolutional block (`layer4`) along with the new output layer. This allowed the model to gently adapt its high-level feature detection to the specific shapes in the CIFAR-10 dataset.

## Training & Evaluation

* **Optimizer:** Adam Optimizer (learning rate = `0.001` for the Custom CNN and Feature Extraction, and `0.0005` for Fine-Tuning).
* **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`).
* **Early Stopping:** Implemented a custom Early Stopping mechanism (patience = 3 epochs) that monitors the validation loss to halt training when the model stops improving, saving the best weights.

### Results

| Model | Accuracy | Approx. Train Time | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **Custom CNN** | **80%** | ~4 mins (GPU) | Great baseline; fast and highly optimized for 32x32 inputs. |
| **ResNet18 (Feature Extraction)** | **81%** | ~18 mins (GPU) | Limited by frozen weights and the dimension mismatch (required upscaling). |
| **ResNet18 (Fine-Tuned)**| **91%** | ~15 mins (GPU)* | **Best Performer.** Unfreezing `layer4` allowed the model to bridge the gap between ImageNet features and CIFAR-10. |

*\*Note: Fine-Tuning triggered Early Stopping faster, resulting in a slightly shorter total training time than the frozen ResNet.*

Detailed classification reports and Confusion Matrices for each model are generated within the notebook. A common trend across all models was a higher confusion rate between visually similar classes, specifically "cats" and "dogs."

## Requirements

To reproduce this project, ensure you have a Python environment with the following packages installed:

```txt
torch
torchvision
matplotlib
numpy
seaborn
scikit-learn
