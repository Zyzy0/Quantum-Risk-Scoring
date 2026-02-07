import pennylane as qml
from pennylane import numpy as np


class QuantumRiskModel:
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Inicjalizacja wag dla warstw StronglyEntanglingLayers
        self.weights = 0.01 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)

        @qml.qnode(self.dev)
        def circuit(weights, data):
            qml.AngleEmbedding(data, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def predict(self, weights, data):
        expectations = self.circuit(weights, data)
        return np.argmax(expectations)

    def cost(self, weights, X, y):
        loss = 0
        for i in range(len(X)):
            predictions = self.circuit(weights, X[i])
            target_onehot = np.zeros(self.n_qubits)
            target_onehot[int(y[i])] = 1
            loss += np.sum((predictions - target_onehot) ** 2)
        return loss / len(X)