import sys
import os
from sklearn.metrics import classification_report

# Dodanie bieżącej ścieżki do sys.path, aby uniknąć błędów importu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.utils import prepare_quantum_data
from src.quantum_model import QuantumRiskModel
from src.classical_baseline import run_classical_baseline
import pennylane as qml
from pennylane import numpy as np


def main():
    # 1. Załadowanie i przygotowanie danych
    print("--- KROK 1: Przygotowywanie danych ---")
    file_path = 'data/features_ml_data.csv'

    try:
        X_train, X_test, y_train, y_test = prepare_quantum_data(file_path, n_components=4)
        print(f"Dane załadowane. Rozmiar treningowy: {X_train.shape}, Testowy: {X_test.shape}")
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Uruchomienie modelu klasycznego (Benchmark)
    print("\n--- KROK 2: Uruchamianie Baseline (XGBoost) ---")
    run_classical_baseline(X_train, X_test, y_train, y_test)

    # 3. Konfiguracja i Trening Modelu Kwantowego
    print("\n--- KROK 3: Trenowanie modelu kwantowego (VQC) ---")
    q_model = QuantumRiskModel(n_qubits=4, n_layers=3)
    opt = qml.AdamOptimizer(stepsize=0.05)  # Mniejszy krok dla lepszej stabilności
    weights = q_model.weights

    # Zwiększamy zakres treningowy dla lepszych wyników (np. 100 próbek)
    # Uwaga: symulacja kwantowa jest wolna, dostosuj wielkość batcha do sprzętu
    batch_size = 32
    epochs = 10

    print(f"Rozpoczynam uczenie: {epochs} epok, batch size: {batch_size}...")

    for epoch in range(epochs):
        # Wybieramy losowy batch dla każdej epoki (Mini-batch GD)
        indices = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]

        # Aktualizacja wag
        weights = opt.step(lambda w: q_model.cost(w, X_batch, y_batch), weights)

        current_loss = q_model.cost(weights, X_batch, y_batch)
        print(f"Epoka {epoch + 1}/{epochs} | Loss: {current_loss:.4f}")

    # 4. Ewaluacja i czyszczenie wyników
    print("\n--- KROK 4: Ewaluacja wyników kwantowych ---")
    test_size = 50  # Sprawdzamy na 50 próbkach testowych

    # Przewidywanie
    q_raw_preds = [q_model.predict(weights, x) for x in X_test[:test_size]]
    q_preds = [int(p) for p in q_raw_preds]
    actuals = [int(a) for a in y_test[:test_size]]

    print("\nPorównanie (pierwsze 20 próbek):")
    print(f"Predykcje Modelu: {q_preds[:20]}")
    print(f"Wartości Realne:  {actuals[:20]}")

    # 5. Prosty raport końcowy
    print("\n--- RAPORT KOŃCOWY ---")
    print(classification_report(actuals, q_preds, target_names=['Low', 'Medium', 'High', 'Very High'],
                                labels=[0, 1, 2, 3]))

    # Zapisz wagi do pliku .npy
    np.save('data/quantum_weights.npy', weights)
    print("Wagi modelu zostały zapisane w data/quantum_weights.npy")


if __name__ == "__main__":
    main()