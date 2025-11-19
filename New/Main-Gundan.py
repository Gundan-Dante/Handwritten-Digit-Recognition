import pandas as pd
import numpy as np

# Load MNIST data
data = pd.read_csv('mnist_train.csv')

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow

def sigmoid_derivative(a):
    """Derivative of sigmoid function"""
    return a * (1 - a)

def initialize_parameters():
    """
    Initialize weights and biases with Xavier initialization
    for each layer because why not
    """
    np.random.seed(42)
    params = {
        'W1': np.random.randn(16, 784) * np.sqrt(2.0 / 784),
        'b1': np.zeros((16, 1)),
        'W2': np.random.randn(16, 16) * np.sqrt(2.0 / 16),
        'b2': np.zeros((16, 1)),
        'W3': np.random.randn(10, 16) * np.sqrt(2.0 / 16),
        'b3': np.zeros((10, 1))
    }
    """
    So we simply create a params dictionary that includes
    weights and biases matrices being values of recpect
    layers for 784 -> 16 -> 16 -> 10 kind of mapping. Taken
    from 3Blue1Brown btw.
    """
    return params

def load_parameters():
    """Load parameters from files"""
    try:
        with open("Parameters.txt", "r") as f:
            weights = [float(line.strip()) for line in f.readlines()]
        
        with open("bias.txt", "r") as f:
            biases = [float(line.strip()) for line in f.readlines()]
        
        params = {
            'W1': np.array(weights[:784*16]).reshape(16, 784),
            'b1': np.array(biases[:16]).reshape(16, 1),
            'W2': np.array(weights[784*16:784*16+16*16]).reshape(16, 16),
            'b2': np.array(biases[16:32]).reshape(16, 1),
            'W3': np.array(weights[784*16+16*16:]).reshape(10, 16),
            'b3': np.array(biases[32:]).reshape(10, 1)
        }
        return params
    except FileNotFoundError:
        print("Parameter files not found. Initializing new parameters.")
        return initialize_parameters()

def save_parameters(params):
    """Save parameters to files"""
    with open("Parameters.txt", "w") as f:
        for w in [params['W1'].flatten(), params['W2'].flatten(), params['W3'].flatten()]:
            for val in w:
                f.write(f"{val}\n")
    
    with open("bias.txt", "w") as f:
        for b in [params['b1'].flatten(), params['b2'].flatten(), params['b3'].flatten()]:
            for val in b:
                f.write(f"{val}\n")

def forward_propagation(X, params):
    """Perform forward propagation"""
    # X is the 784 input values
    cache = {} # To save intermediate results for backward propagation
    
    # Layer 1
    cache['Z1'] = params['W1'] @ X + params['b1']
    cache['A1'] = sigmoid(cache['Z1'])
    
    # Layer 2
    cache['Z2'] = params['W2'] @ cache['A1'] + params['b2']
    cache['A2'] = sigmoid(cache['Z2'])
    
    # Layer 3 (output)
    cache['Z3'] = params['W3'] @ cache['A2'] + params['b3']
    cache['A3'] = sigmoid(cache['Z3'])
    
    return cache

def compute_cost(A3, y):
    """Compute mean squared error cost"""
    m = y.shape[1]
    cost = np.sum((A3 - y) ** 2) / (2 * m)
    return cost

def backward_propagation(X, y, params, cache):
    """Perform backpropagation to compute gradients"""
    m = X.shape[1]
    
    # Output layer gradients
    dZ3 = (cache['A3'] - y) * sigmoid_derivative(cache['A3'])
    dW3 = (dZ3 @ cache['A2'].T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    
    # Hidden layer 2 gradients
    dA2 = params['W3'].T @ dZ3
    dZ2 = dA2 * sigmoid_derivative(cache['A2'])
    dW2 = (dZ2 @ cache['A1'].T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    # Hidden layer 1 gradients
    dA1 = params['W2'].T @ dZ2
    dZ1 = dA1 * sigmoid_derivative(cache['A1'])
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {
        'dW1': dW1, 'db1': db1,
        'dW2': dW2, 'db2': db2,
        'dW3': dW3, 'db3': db3
    }
    
    return gradients

def update_parameters(params, gradients, learning_rate):
    """Update parameters using gradient descent"""
    params['W1'] -= learning_rate * gradients['dW1']
    params['b1'] -= learning_rate * gradients['db1']
    params['W2'] -= learning_rate * gradients['dW2']
    params['b2'] -= learning_rate * gradients['db2']
    params['W3'] -= learning_rate * gradients['dW3']
    params['b3'] -= learning_rate * gradients['db3']
    
    return params

def train(data, epochs=10, learning_rate=0.1, batch_size=32):
    """Train the neural network"""
    params = load_parameters()
    
    for epoch in range(epochs):
        # Shuffle data
        shuffled_data = data.sample(frac=1).reset_index(drop=True)
        total_cost = 0
        num_batches = len(shuffled_data) // batch_size
        
        for i in range(num_batches):
            # Get batch
            batch = shuffled_data.iloc[i*batch_size:(i+1)*batch_size]
            
            # Prepare input (normalize pixel values)
            X = batch.iloc[:, 1:].values.T / 255.0
            
            # Prepare labels (one-hot encoding)
            labels = batch.iloc[:, 0].values
            y = np.zeros((10, batch_size))
            for j, label in enumerate(labels):
                y[label, j] = 1
            
            # Forward propagation
            cache = forward_propagation(X, params)
            
            # Compute cost
            cost = compute_cost(cache['A3'], y)
            total_cost += cost
            
            # Backward propagation
            gradients = backward_propagation(X, y, params, cache)
            
            # Update parameters
            params = update_parameters(params, gradients, learning_rate)
        
        avg_cost = total_cost / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Cost: {avg_cost:.6f}")
        
        # Save parameters after each epoch
        save_parameters(params)
        with open("cost.txt", "w") as f:
            f.write(str(avg_cost))
    
    return params

def predict(X, params):
    """Make predictions"""
    cache = forward_propagation(X, params)
    predictions = np.argmax(cache['A3'], axis=0)
    return predictions

def test_accuracy(test_data, params):
    """Test the model accuracy"""
    X_test = test_data.iloc[:, 1:].values.T / 255.0
    y_test = test_data.iloc[:, 0].values
    
    predictions = predict(X_test, params)
    accuracy = np.mean(predictions == y_test) * 100
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    return accuracy

# Example usage
if __name__ == "__main__":
    print("Training Neural Network on MNIST dataset...")
    print("Architecture: 784 -> 16 -> 16 -> 10")
    print("-" * 50)
    
    # Train the model
    params = train(data, epochs=10, learning_rate=0.5, batch_size=32)
    
    # Test on a small subset (if you have test data)
    test_data = pd.read_csv('mnist_test.csv')
    test_accuracy(test_data, params)