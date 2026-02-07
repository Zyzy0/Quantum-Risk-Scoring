import pennylane as qml
from pennylane import numpy as np


class QuantumRiskModel:
    def __init__(self, n_qubits=4, n_layers=4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = 0.1 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)
        self.bias = np.zeros(n_qubits, requires_grad=True)

        @qml.qnode(self.dev)
        def circuit(weights, data):
            qml.AngleEmbedding(data, wires=range(self.n_qubits), rotation='X')
            qml.AngleEmbedding(data, wires=range(self.n_qubits), rotation='Y')

            for l in range(len(weights)):
                for i in range(self.n_qubits):
                    qml.Rot(*weights[l, i], wires=i)
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def predict(self, weights, bias, data):
        expectations = np.array(self.circuit(weights, data)) + bias
        probs = np.exp(expectations) / np.sum(np.exp(expectations))
        return np.argmax(probs)

    def cost(self, params, X, y):
        weights, bias = params
        loss = 0
        for i in range(len(X)):
            expectations = np.array(self.circuit(weights, X[i])) + bias
            probs = np.exp(expectations) / np.sum(np.exp(expectations))

            target_idx = int(y[i])
            loss -= np.log(probs[target_idx] + 1e-9)

        return loss / len(X)