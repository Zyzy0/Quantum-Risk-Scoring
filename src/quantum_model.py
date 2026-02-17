import pennylane as qml
from pennylane import numpy as np


class QuantumRiskModel:
    def __init__(self, n_qubits=8, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Penalty weights for classes [Low, Med, High, V.High]
        self.class_weights = np.array([1.0, 1.0, 3.0, 8.0], requires_grad=False)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, data):
            # Data Re-uploading strategy
            for l in range(self.n_layers):
                qml.AngleEmbedding(data, wires=range(self.n_qubits), rotation='X')

                # Use weights passed as an argument
                # Slicing [l:l+1] maintains the correct dimensions for Pennylane
                qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(self.n_qubits))

            # Return probabilities for the first 2 qubits (representing 4 basis states)
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def predict(self, weights, bias, data):
        # Bias is ignored (None), but the argument is kept for compatibility
        probs = self.circuit(weights, data)
        return np.argmax(probs)

    def cost(self, weights, X, y):
        loss = 0.0
        for i in range(len(X)):
            probs = self.circuit(weights, X[i])
            target_idx = int(y[i])

            # Retrieve the penalty weight for this specific class
            w_class = self.class_weights[target_idx]

            # Weighted Cross-Entropy Loss
            loss = loss - (w_class * np.log(probs[target_idx] + 1e-9))

        return loss / len(X)