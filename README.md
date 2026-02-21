Custom Neural Network from Scratch: MNIST Digit Classification

This repository contains a complete, ground-up implementation of deep learning components and a Multilayer Perceptron (MLP) built entirely in Python using NumPy. High-level deep learning frameworks (like TensorFlow or PyTorch) are avoided to demonstrate the core mathematics, forward propagation, and backpropagation mechanics behind neural networks.

Repository Structure
* `model.py` / `cnn-from-scratch.ipynb`: Contains the core implementations for data preprocessing, custom neural network layers, training loop, and evaluation.
* `model.pkl`: Saved weights and biases of the trained network (custom serialization).
* `deploy.py`: Script containing the logic for model deployment and inference.

Problem Statement
The objective of this project is to build a mathematical deep learning model from scratch to accurately recognize and classify handwritten digits (0-9). By implementing algorithms like gradient descent and backpropagation manually, this project showcases the fundamental workings of neural networks in computer vision tasks.

Dataset Description
The model is trained and evaluated on the classic **MNIST Handwritten Digit Dataset**.
* **Features:** 28x28 pixel grayscale images, flattened into 1D arrays of size 784.
* **Pixel Range:** Normalized from `[0, 255]` to `[0, 1]` (`float32`).
* **Labels:** Integers from 0 to 9, converted into one-hot encoded vectors.
* **Split:** 60,000 images in the training set and 10,000 images in the test set.

Model Architecture Explanation
The repository features object-oriented implementations of Convolutional Layers (`ConvLayer`), Pooling Layers (`MaxPoolLayer`), and standard Dense Layers. 

The primary model trained in this project is a **4-Layer Multilayer Perceptron (MLP)** with the following architecture:
1. **Input Layer:** 784 neurons (flattened 28x28 images).
2. **Hidden Layer 1:** 128 neurons, activated by **ReLU**.
3. **Hidden Layer 2:** 64 neurons, activated by **ReLU**.
4. **Output Layer:** 10 neurons, activated by **Softmax** (to output probabilities for the 10 digit classes).

Implementation Details
* **Zero-Framework Dependencies:** The core math relies exclusively on `NumPy`.
* **Weight Initialization:** Uses Xavier/He initialization to maintain variance across layers and prevent vanishing/exploding gradients.
* **Optimization Algorithm:** Mini-batch Gradient Descent with a batch size of 128.
* **Learning Rate Decay:** Starts at `0.01` and decays by a factor of `0.95` every epoch to ensure smooth convergence.
* **Stability Safeguards:** Implements gradient clipping (limits $dZ$ between -10 and 10, and gradients between -5 and 5) and input clipping for numerically stable Softmax and Sigmoid operations.
* **Data Preprocessing:** Fetched via `scikit-learn`, scaled using standard division, and one-hot encoded via `LabelBinarizer`.

Evaluation Metrics Used
The performance of the model is monitored using:
1. **Categorical Cross-Entropy Loss:** Calculates the logarithmic loss between the predicted probabilities and the true one-hot encoded labels.
2. **Accuracy:** The percentage of correctly predicted images out of the total images in the batch/dataset (Tracks both Training Accuracy and Test Accuracy). 
*The model achieves a final Test Accuracy of ~94.7%.*
