import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Configuration
DATA_PATH = 'data/final_ml_dataset.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_global_model.pkl')

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please run data generation scripts first.")
    
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure Date is datetime for sorting
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'Ticker'])
    
    return df

def train_model():
    df = load_data()
    
    # Updated Feature List including Quarterly & Tech Indicators
    features = [
        'Log_Return', 'RSI', 'Volat_20d', 'SMA_20_Ratio', 'Vol_Change', 'Rel_Return', 'Market_Corr',
        'SMA_50_Ratio', 'Return_90d', 'Volat_90d', 'Drawdown_90d', 'Bollinger_Pct', 'Bollinger_Width',
        'MARKET_Log_Return'
    ]
    target = 'TARGET_Risk_Level'
    
    # --- TEMPORAL SPLIT ---
    dates = df['Date'].unique()
    split_idx = int(len(dates) * 0.8) # Last 20% for testing
    split_date = dates[split_idx]
    
    print(f"Splitting data at date: {split_date}")
    
    train_mask = df['Date'] < split_date
    test_mask = df['Date'] >= split_date
    
    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target]
    
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, target]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # --- HYPERPARAMETER TUNING (Grid Search) ---
    print("\nStarting Hyperparameter Tuning (GridSearchCV)...")
    print("This may take a few minutes...")
    
    # We use TimeSeriesSplit for validation within the training set
    tscv = TimeSeriesSplit(n_splits=3)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_leaf': [2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    best_rf = grid_search.best_estimator_
    
    # --- EVALUATION ---
    print("\nEvaluating Best Model on Test Set...")
    y_pred = best_rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy (Test Set): {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature Importances:")
    for f in range(len(features)):
        print(f"{features[indices[f]]}: {importances[indices[f]]:.4f}")

    # --- SAVE METRICS ---
    RESULTS_DIR = 'results'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    metrics_path = os.path.join(RESULTS_DIR, 'rf_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"RF Model Accuracy: {acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nFeature Importances:\n")
        for i in range(len(features)):
             f.write(f"{features[indices[i]]}: {importances[indices[i]]:.4f}\n")
    
    print(f"Metrics saved to {metrics_path}")

    # --- SAVE MODEL ---
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(best_rf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return best_rf, features

def predict_latest(ticker):
    """
    Loads the trained model and predicts risk for the latest available data point for a specific ticker.
    """
    # Features MUST match the training list order
    features = [
        'Log_Return', 'RSI', 'Volat_20d', 'SMA_20_Ratio', 'Vol_Change', 'Rel_Return', 'Market_Corr',
        'SMA_50_Ratio', 'Return_90d', 'Volat_90d', 'Drawdown_90d', 'Bollinger_Pct', 'Bollinger_Width',
        'MARKET_Log_Return'
    ]

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training new model...")
        rf, _ = train_model()
    else:
        print(f"Loading model from {MODEL_PATH}...")
        rf = joblib.load(MODEL_PATH)

    df = load_data()
    
    # Filter for ticker
    ticker_df = df[df['Ticker'] == ticker].copy()
    
    if ticker_df.empty:
        print(f"No data found for ticker: {ticker}")
        return

    # Get latest data point
    latest_row = ticker_df.iloc[-1]
    latest_date = latest_row['Date']
    
    # Prepare features for prediction (single sample)
    X_new = ticker_df.iloc[[-1]][features]
    
    prediction = rf.predict(X_new)[0]
    probabilities = rf.predict_proba(X_new)[0]
    
    risk_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    predicted_label = risk_labels.get(prediction, "Unknown")
    
    print(f"\n--- Prediction for {ticker} on {latest_date.date()} ---")
    print(f"Predicted Risk Level: {predicted_label}")
    print(f"Confidence (Probabilities):")
    for i, label in risk_labels.items():
        if i < len(probabilities):
            print(f"  {label}: {probabilities[i]:.2%}")
            
    return predicted_label

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    else:
        # Ask user if they want to retrain if run manually
        # For now, we assume if run directly we might want to test prediction
        pass
    
    predict_latest("AAPL")
    predict_latest("TSLA")
