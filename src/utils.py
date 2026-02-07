import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def prepare_quantum_data(file_path, n_components=4):
    df = pd.read_csv(file_path, index_col=0)

    # Zakładamy, że ostatnia kolumna to Target (ryzyko 0-3)
    # Jeśli jeszcze nie masz etykiet, musisz je wygenerować w preprocessingu
    if 'Target' not in df.columns:
        # Przykładowe generowanie etykiet na podstawie zmienności, jeśli ich brak
        vol_col = [c for c in df.columns if 'Volat' in c][0]
        df['Target'] = pd.qcut(df[vol_col], 4, labels=[0, 1, 2, 3]).astype(int)

    # Podział chronologiczny (80% trening, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df.drop('Target', axis=1)
    y_train = train_df['Target'].values
    X_test = test_df.drop('Target', axis=1)
    y_test = test_df['Target'].values

    # Skalowanie i PCA
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(scaler.fit_transform(X_train))
    X_test_pca = pca.transform(scaler.transform(X_test))

    # Normalizacja do zakresu [0, pi] dla AngleEmbedding
    X_train_pca = np.pi * (X_train_pca - X_train_pca.min()) / (X_train_pca.max() - X_train_pca.min())
    X_test_pca = np.pi * (X_test_pca - X_test_pca.min()) / (X_test_pca.max() - X_test_pca.min())

    return X_train_pca, X_test_pca, y_train, y_test