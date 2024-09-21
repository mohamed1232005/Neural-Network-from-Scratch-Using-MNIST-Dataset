# Neural-Network-from-Scratch-Using-MNIST-Dataset

1) ## Project Overview:

This project focuses on building a neural network from scratch, aiming to classify handwritten digits from the MNIST dataset. Rather than using high-level machine learning libraries such as TensorFlow or PyTorch, the network is developed using only basic Python libraries like NumPy. This approach demonstrates a deep understanding of how neural networks operate at a low level, including key processes such as forward propagation, backpropagation, and optimization.

### Dataset :

The dataset used is the MNIST dataset, a benchmark dataset for image classification tasks. It consists of 60,000 training examples and 10,000 test examples, with each example being a grayscale image of a digit from 0 to 9, represented in a 28x28 pixel grid. To simplify the input, each image was flattened into a one-dimensional vector of 784 pixels.

**Key Steps in Data Preparation:**
Normalization: To ensure the neural network trains effectively, the pixel values (originally ranging from 0 to 255) were scaled down to a range of 0 to 1.
One-Hot Encoding: The labels (digits 0-9) were transformed into a binary format where each digit is represented by a vector of length 10, where one element is set to 1 (indicating the digit) and the others are set to 0.


### **Neural Network Architecture** : 

The neural network architecture implemented in this project consists of:

-Input Layer: This layer consists of 784 neurons, one for each pixel in the flattened input images.
-Hidden Layer: The hidden layer has 532 neurons. This layer applies the ReLU (Rectified Linear Unit) activation function to introduce non-linearity, allowing the network to learn complex patterns in the data.
Output Layer: The output layer consists of 10 neurons, corresponding to the 10 possible digit classes (0-9). The softmax function is applied to the output layer to convert the raw scores into probabilities.

3- ### **Training Process**: 

