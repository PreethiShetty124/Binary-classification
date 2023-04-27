# Binary-classification - Horse or Human images

# Objective

This project is done with the help of DeepLearning,AI, TensorFlow. This project covers 4 courses:
Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
Convolutional Neural Networks in TensorFlow
Natural Language Processing in TensorFlow
Sequences, Time Series and Prediction

Horses or Humans is a dataset of 300×300 images, created by Laurence Moroney, that is licensed CC-By-2.0 for anybody to use in learning or testing computer vision algorithms.

The objective of this study is to correctly identify if the image is a horse or a human.

# Code and Resources Used
Phyton Version: 3.0
Packages: pandas, numpy, sklearn, seaborn, matplotlib, tensorflow, keras, os.

# Data Description
The set contains 500 rendered images of various species of horse in various poses in various locations. It also contains 527 rendered images of humans in various poses and locations. Emphasis has been taken to ensure diversity of humans, and to that end there are both men and women as well as Asian, Black, South Asian and Caucasians present in the training set. The validation set adds 6 different figures of different gender, race and pose to ensure breadth of data.

# Feature engineering
We define each directory using os library.

Use of data generators. It read the pictures in our source folders. We have one generator for the training images and one for the validation images. The two generators yield batches of images of size 300x300 and their labels (binary).

Data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. In our case, we will preprocess our images by normalizing the pixel values to be in the [0, 1] range (originally all values are in the [0, 255] range). In Keras this can be done via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you to instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can then be used with the Keras model methods that accept data generators as inputs: fit, evaluate_generator, and predict_generator.

# Neural Network with convolution and pooling
This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the following:

Three Convolution layer with a MaxPooling layer which is then designed to compress the image, while maintaining the content of the features that were highlighted by the convlution. By specifying (3,3) for the MaxPooling, the effect is to quarter the size of the image.
One flatten layer: It turns the images into a 1 dimensional set.
Two Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer consisted in 512 neurons with relu as an activation function. The second, have 1 neurons and sigmoid as activation function.
We built this model using adam optimizer and binary_crossentropy as loss function.

The number of epochs=15

We obtained Accuracy 1.00 for the train data and Accuracy 0.7773 for the validation data.

