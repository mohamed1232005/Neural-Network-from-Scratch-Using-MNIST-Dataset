# Neural-Network-from-Scratch-Using-MNIST-Dataset

## Project Overview:

This project focuses on building a neural network from scratch, aiming to classify handwritten digits from the MNIST dataset. Rather than using high-level machine learning libraries such as TensorFlow or PyTorch, the network is developed using only basic Python libraries like NumPy. This approach demonstrates a deep understanding of how neural networks operate at a low level, including key processes such as forward propagation, backpropagation, and optimization.

 ###  **1- Dataset** :

The dataset used is the MNIST dataset, a benchmark dataset for image classification tasks. It consists of 60,000 training examples and 10,000 test examples, with each example being a grayscale image of a digit from 0 to 9, represented in a 28x28 pixel grid. To simplify the input, each image was flattened into a one-dimensional vector of 784 pixels.

**Key Steps in Data Preparation:**
- **Normalization**: To ensure the neural network trains effectively, the pixel values (originally ranging from 0 to 255) were scaled down to a range of 0 to 1.
- **One-Hot Encoding**: The labels (digits 0-9) were transformed into a binary format where each digit is represented by a vector of length 10, where one element is set to 1 (indicating the digit) and the others are set to 0.


 ### **2- Neural Network Architecture** : 

The neural network architecture implemented in this project consists of:

- **Input Layer**: This layer consists of 784 neurons, one for each pixel in the flattened input images.
- **Hidden Layer**: The hidden layer has 532 neurons. This layer applies the ReLU (Rectified Linear Unit) activation function to introduce non-linearity, allowing the network to learn complex patterns in the data.
- **Output Layer**: The output layer consists of 10 neurons, corresponding to the 10 possible digit classes (0-9). The softmax function is applied to the output layer to convert the raw scores into probabilities.

 ### **3- Training Process**:

 The neural network was trained using supervised learning, where the goal was to minimize the difference between the predicted outputs and the actual labels (digits). The key aspects of the training process included:

- **Forward Propagation:** During forward propagation, the input data was passed through the network layers, producing predictions. These predictions were then compared to the actual labels to compute the error (or loss).
- **Backpropagation:** Once the error was calculated, backpropagation was used to update the network’s weights. This process calculates how much each weight contributed to the error and adjusts it to reduce the error in future predictions.
- **Optimization:** The weights were updated using gradient descent, an optimization algorithm that aims to minimize the loss function by iteratively adjusting the weights in the direction that reduces the error.

The training was performed over 250 epochs (iterations), with a learning rate of 0.03, controlling how quickly the model updates its weights.

### **4- Model Evaluation**:
After training, the model was evaluated on the test set. The test set contains examples the network has not seen during training, providing a good measure of the model's ability to generalize. The key performance metrics were:

- **Accuracy**: The model achieved a high classification accuracy on the test set, which indicates that it learned to classify the digits correctly in most cases.
- **Confusion Matrix**: A confusion matrix was generated to show how often the model predicted each digit correctly and where it made mistakes. This matrix is particularly useful for identifying digits the model struggles to classify.

### **4- Results** :
- **Accuracy: ** The neural network achieved an accuracy of 89% on the test set. While this is a decent performance for a neural network built from scratch, there is still room for improvement through potential enhancements like tuning hyperparameters, adjusting the network architecture, or increasing the number of training epochs.
- **Loss: ** Over the course of the training process, the loss function decreased consistently, indicating that the model was learning effectively and improving its predictions.
- **Confusion Matrix: ** A confusion matrix was generated to provide a more detailed breakdown of the model’s performance. It highlights how often the model predicted each digit correctly and where it misclassified digits. This can help identify patterns in misclassifications and further guide optimization.
