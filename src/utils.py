import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def prepare_quantum_data(file_path, n_components=8):
    df = pd.read_csv(file_path, index_col=0)

    if 'TARGET_Risk_Level' not in df.columns:
        vol_col = [c for c in df.columns if 'Volat' in c][0]
        df['Target'] = pd.qcut(df[vol_col], 4, labels=[0, 1, 2, 3]).astype(int)
    else:
        df['Target'] = df['TARGET_Risk_Level']

    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    drop_cols = ['Target', 'Date', 'Ticker', 'TARGET_Future_Vol', 'TARGET_Risk_Level']
    X_train = train_df.drop(drop_cols, axis=1, errors='ignore')
    y_train = train_df['Target'].values
    X_test = test_df.drop(drop_cols, axis=1, errors='ignore')
    y_test = test_df['Target'].values

    # 1. Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. PCA (8 components)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 3. MinMax [0, PI] (For AngleEmbedding)
    minmax = MinMaxScaler(feature_range=(0, np.pi))
    X_train_final = minmax.fit_transform(X_train_pca)
    X_test_final = minmax.transform(X_test_pca)

    return X_train_final, X_test_final, y_train, y_test