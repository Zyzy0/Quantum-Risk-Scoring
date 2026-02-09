import sys
import os
from sklearn.metrics import classification_report, accuracy_score
import pennylane as qml
from pennylane import numpy as np

# Adjust path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.utils import prepare_quantum_data
from src.quantum_model import QuantumRiskModel

def main():
    print("--- STEP 1: Data Preparation (8 Components) ---")
    file_path = 'data/features_ml_data.csv'
    X_train, X_test, y_train, y_test = prepare_quantum_data(file_path, n_components=8)
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")

    print("\n--- STEP 2: Training with Model Checkpoint ---")
    q_model = QuantumRiskModel(n_qubits=8, n_layers=3)
    params = q_model.weights

    # Using a smaller learning rate for stability
    opt = qml.AdamOptimizer(stepsize=0.02)
    batch_size = 32
    epochs = 60

    # Best model tracking
    best_acc = 0.0
    best_weights = params
    best_epoch = 0

    print(f"Starting: {epochs} epochs...")

    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        batch_losses = []
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i: i + batch_size]
            y_batch = y_train[i: i + batch_size]

            params, loss = opt.step_and_cost(lambda w: q_model.cost(w, X_batch, y_batch), params)
            batch_losses.append(loss)

        avg_loss = np.mean(batch_losses)

        # Validation per epoch
        val_sample_size = min(100, len(X_test))
        test_indices = np.random.choice(len(X_test), val_sample_size, replace=False)
        preds = [q_model.predict(params, None, X_test[i]) for i in test_indices]
        current_acc = accuracy_score(y_test[test_indices], preds)

        # Checkpoint mechanism
        if current_acc > best_acc:
            best_acc = current_acc
            best_weights = params.copy()
            best_epoch = epoch + 1
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | Acc: {current_acc:.2%} (*) NEW BEST")
        else:
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | Acc: {current_acc:.2%}")

    print(f"\n--- Training Complete ---")
    print(f"Loading weights from Epoch {best_epoch} (Best Acc: {best_acc:.2%})")

    # Save the best parameters
    quantum_payload = {'weights': best_weights, 'bias': 0}
    np.save('data/quantum_weights.npy', quantum_payload, allow_pickle=True)
    print("Best weights saved successfully.")

    print("\n--- Final Classification Report (Best Model) ---")
    # Evaluate on a larger test subset
    test_limit = 300
    final_preds = [q_model.predict(best_weights, None, x) for x in X_test[:test_limit]]
    print(classification_report(y_test[:test_limit], final_preds,
                                target_names=['Low', 'Med', 'High', 'V.High'],
                                zero_division=0))

if __name__ == "__main__":
    main()