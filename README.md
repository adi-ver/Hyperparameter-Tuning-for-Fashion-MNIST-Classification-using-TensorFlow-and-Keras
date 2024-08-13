# Hyperparameter-Tuning-for-Fashion-MNIST-Classification-using-TensorFlow-and-Keras
This project focuses on hyperparameter tuning for a neural network model that classifies images from the Fashion MNIST dataset. The tuning process is conducted using Keras Tuner with Bayesian Optimization to find the best combination of hyperparameters for the model.

### Introduction
Hyperparameter tuning is a crucial step in optimizing the performance of machine learning models. This project demonstrates how to use TensorFlow and Keras Tuner to perform hyperparameter tuning on a neural network designed to classify images from the Fashion MNIST dataset.

### Dataset
The Fashion MNIST dataset consists of 60,000 training images and 10,000 test images, each being a 28x28 grayscale image associated with a label from one of 10 classes. The dataset is a collection of fashion items such as shoes, t-shirts, and bags.

### Model Architecture
The model is a Sequential neural network consisting of:
- Flatten Layer: Converts the 2D image into a 1D array.
- Lambda Layer: Normalizes the pixel values by dividing by 255.
- Dense Layers: One or more dense layers with ReLU activation.
- Dropout Layers: Dropout is applied after each dense layer to prevent overfitting.
- Output Layer: A dense layer with 10 units and softmax activation to classify the images into one of the 10 classes.

### Hyperparameter Tuning
The following hyperparameters were tuned using Keras Tuner with Bayesian Optimization:
- Number of Hidden Layers: 1, 2, or 3.
- Number of Units in Dense Layers: 8, 16, or 32.
- Dropout Rate: Between 0.1 and 0.5.
- Learning Rate: Between 0.0001 and 0.01.
- Batch Size: Between 32 and 128.
- The custom tuner was created by subclassing the kerastuner.tuners.BayesianOptimization class to include the batch size in the tuning process.

### Training
The model was trained using the Adam optimizer with sparse categorical crossentropy as the loss function. The tuning process was conducted over 20 trials, and the best model was selected based on validation accuracy. Further training was done using the best model with an early stopping callback to prevent overfitting.

### Evaluation
The model's performance was evaluated based on accuracy and loss metrics on both the training and validation datasets. The early stopping callback was set to monitor the validation accuracy, stopping the training if it did not improve for 3 consecutive epochs.

### Usage
To replicate this project:

Install the required libraries, including TensorFlow and Keras Tuner.
Load the Fashion MNIST dataset using tf.keras.datasets.fashion_mnist.
Define the model architecture and hyperparameter search space.
Run the hyperparameter tuning using the CustomTuner.
Train the model using the best hyperparameters.

### Results
The project demonstrates the importance of hyperparameter tuning in improving model performance. The final model achieved a significant level of accuracy on the Fashion MNIST dataset.
