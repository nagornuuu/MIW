import numpy as np


def load_data(file_path):
    """
    Load data from a given file and convert it to numpy arrays.
    The file is expected to contain two columns: input and output.

    Args:
    file_path (str): The path to the dataset file.

    Returns:
    np.ndarray: Numpy array with inputs.
    np.ndarray: Numpy array with outputs.
    """
    data = np.loadtxt('C:/Users/andrii/Desktop/MIWczwiczenia/miw04/dane/dane1.txt')
    inputs = data[:, 0]
    outputs = data[:, 1]
    return inputs, outputs


def split_data(inputs, outputs, train_ratio=0.8):
    """
    Split the inputs and outputs into training and testing datasets based on the given ratio.

    Args:
    inputs (np.ndarray): The input data.
    outputs (np.ndarray): The output data.
    train_ratio (float): The ratio of data to use for training (default is 0.8).

    Returns:
    tuple: Tuple containing training inputs, training outputs, testing inputs, testing outputs.
    """
    # Generate indices and shuffle them
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)

    # Split indices
    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create training and testing sets
    train_inputs = inputs[train_indices]
    train_outputs = outputs[train_indices]
    test_inputs = inputs[test_indices]
    test_outputs = outputs[test_indices]

    return train_inputs, train_outputs, test_inputs, test_outputs

# Example usage:
inputs, outputs = load_data("daneXX.txt")
train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)
def sigmoid(x):
    """
    Sigmoid activation function.

    Args:
    x (np.ndarray): Input array or scalar to the function.

    Returns:
    np.ndarray: Output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Hyperbolic tangent activation function.

    Args:
    x (np.ndarray): Input array or scalar to the function.

    Returns:
    np.ndarray: Output of the tanh function.
    """
    return np.tanh(x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        """
        Initialize the neural network with one hidden layer.

        Args:
        input_size (int): Number of input neurons.
        hidden_size (int): Number of hidden neurons.
        output_size (int): Number of output neurons.
        activation (str): Type of activation function to use ('sigmoid' or 'tanh').
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_func = sigmoid if activation == 'sigmoid' else tanh

        # Weights initialization
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01

    def forward(self, inputs):
        """
        Forward pass through the network.

        Args:
        inputs (np.ndarray): Input data (batch_size x input_size).

        Returns:
        np.ndarray: Output from the network.
        """
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_layer_output = self.activation_func(self.hidden_layer_input)
        self.final_output = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        return self.final_output

    def compute_loss(self, predicted, actual):
        """
        Compute mean squared error loss.

        Args:
        predicted (np.ndarray): Predicted outputs.
        actual (np.ndarray): Actual outputs.

        Returns:
        float: Mean squared error loss.
        """
        return ((predicted - actual) ** 2).mean()

# Example usage:
nn = NeuralNetwork(input_size=1, hidden_size=5, output_size=1, activation='tanh')
outputs = nn.forward(np.array([[0.1], [0.2]]))  # Example input
loss = nn.compute_loss(outputs, np.array([[0.3], [0.2]]))  # Example target


def backpropagation(self, inputs, targets, learning_rate=0.01):
    """
    Perform backpropagation and update the weights.

    Args:
    inputs (np.ndarray): Input data (batch_size x input_size).
    targets (np.ndarray): Actual output data (batch_size x output_size).
    learning_rate (float): Learning rate for the weight update.
    """
    # Forward pass
    outputs = self.forward(inputs)

    # Calculate output layer error (prediction error)
    error_output_layer = outputs - targets

    # Calculate gradients for output layer weights
    gradients_output = np.dot(self.hidden_layer_output.T, error_output_layer)

    # Calculate hidden layer error
    error_hidden_layer = np.dot(error_output_layer, self.weights_hidden_output.T) * \
                         (1 - self.hidden_layer_output ** 2)  # derivative of tanh

    # Calculate gradients for input layer weights
    gradients_input = np.dot(inputs.T, error_hidden_layer)

    # Update weights
    self.weights_hidden_output -= learning_rate * gradients_output
    self.weights_input_hidden -= learning_rate * gradients_input


# Attach the backpropagation method to the NeuralNetwork class
setattr(NeuralNetwork, "backpropagation", backpropagation)

# Example of how to perform backpropagation:
nn = NeuralNetwork(input_size=1, hidden_size=5, output_size=1, activation='tanh')
nn.backpropagation(np.array([[0.1]]), np.array([[0.3]]), learning_rate=0.01)



