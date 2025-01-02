from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np

# create a class of MLP from tensorflow 
def mlp(shape_inp, units, dropout):
        inp = Input(shape=(shape_inp,))
        
        # apply 1st hidden-layer
        x1 = Dense(units=units, activation='relu')(inp)
        x1 = Dropout(dropout)(x1)
        
        # apply 2nd hidden-layer
        x2 = Dense(units=units, activation='relu')(x1)
        x2 = Dropout(dropout)(x2)
        
        # apply Dense + sigmoid
        output = Dense(units = 1, activation='sigmoid')(x2)
        
        # define input and output of the model
        model = Model(inp, output)
        return model
        
        
# create a class of MLP from scratch 
class mlp_scratch:
            
    def __init__(self, input_size, hidden_layers, output_size, function):
        """
        Initialize the MLP model.
        """
        self.layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            biases =  np.random.randn(1, layer_sizes[i + 1]) * 0.1
            if i < len(layer_sizes) - 2:
                self.layers.append({'weights': weights, 'biases': biases, 'activation': function})
            else:
                self.layers.append({'weights': weights, 'biases': biases, 'activation': 'sigmoid'})
                
    def activation(self, z, f):
        if f == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif f == 'relu':
            return np.maximum(0, z)
        
    def activation_der(self, z, f):    
        if f == 'sigmoid':
            return z * (1 - z)
        elif f == 'relu':
            return np.where(z > 0, 1, 0)
        
    def forward(self, X):
        """
        Perform a forward pass through the network.
        """
        activations = [X]
        for i, layer in enumerate(self.layers):
            Z = np.dot(activations[-1], layer['weights']) + layer['biases']
            A = self.activation(Z, layer['activation'])
            activations.append(A)
        return activations
    
    def backward(self, X, y, activations, learning_rate=0.01):
        """
        Perform a backward pass and update weights.
        """
        # Initialize the gradient for the output layer
        output_activation = activations[-1]
        output_delta = (output_activation - y) * self.activation_der(output_activation, self.layers[-1]['activation'])
        deltas = [output_delta]
        
        # Calculate deltas for each layer going backwards
        for i in range(len(self.layers) - 1, 0, -1):
            delta = np.dot(deltas[0], self.layers[i]['weights'].T) * self.activation_der(activations[i], self.layers[i-1]['activation'])
            deltas.insert(0, delta) # insert in the first place (0 position) of the list
        
        # Update weights and biases
        for i, layer in enumerate(self.layers):
            layer['weights'] =  layer['weights'] - learning_rate * np.dot(activations[i].T, deltas[i])
            layer['biases'] = layer['biases'] - learning_rate * np.sum(deltas[i], axis=0, keepdims=True)
    
    def predict(self, X):
        """
        Predict output for given input data.
        """
        return self.forward(X)[-1]
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Train the MLP using gradient descent.
        """
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(X, y, activations, learning_rate)
            
            loss = -np.mean(y * np.log(activations[-1]) + (1 - y) * np.log(1 - activations[-1]))
            print(f'Epoch {epoch}, Loss: {loss}')        
