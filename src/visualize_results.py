import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix
import sys

# Configuration
DATA_PATH = 'data/final_ml_dataset.csv'
MODEL_PATH = 'models/rf_global_model.pkl'
OUTPUT_DIR = 'results'

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Error: Model or Data not found.")
        sys.exit(1)

    print("Loading model and data...")
    rf = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort
    df = df.sort_values(by=['Date', 'Ticker'])
    return rf, df

def plot_feature_importance(rf, feature_names):
    print("Generating Feature Importance Plot...")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title('Random Forest Feature Importance (Global Model)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/feature_importance.png')
    print(f"Saved: {OUTPUT_DIR}/feature_importance.png")
    plt.close()

def plot_confusion_matrix(rf, df, features):
    print("Generating Confusion Matrix...")
    # We need to recreate the test set (last 20% of time)
    dates = df['Date'].unique()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    
    test_mask = df['Date'] >= split_date
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, 'TARGET_Risk_Level']
    
    y_pred = rf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    labels = ['Low', 'Medium', 'High', 'Very High']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png')
    print(f"Saved: {OUTPUT_DIR}/confusion_matrix.png")
    plt.close()

def plot_ticker_risk(df, ticker, rf, features):
    print(f"Generating Risk Plot for {ticker}...")
    
    ticker_df = df[df['Ticker'] == ticker].copy().sort_values(by='Date')
    
    if ticker_df.empty:
        print(f"No data for {ticker}")
        return

    # Predict for the whole history (to see how model would have behaved)
    # Note: In real life, prediction on training data is "cheating", but for visualization
    # of "what the model sees", it's useful.
    X = ticker_df[features]
    predictions = rf.predict(X)
    
    ticker_df['Predicted_Risk'] = predictions
    
    # We need Price data. We don't have raw price in the ML dataset easily available 
    # (we have returns and ratios).
    # However, we can infer a "Price Index" from Log Returns.
    # Start at 100.
    ticker_df['Implied_Price'] = 100 * np.exp(ticker_df['Log_Return'].cumsum())
    
    # Plot
    plt.figure(figsize=(14, 7))
    
    # Plot Price Line
    sns.lineplot(data=ticker_df, x='Date', y='Implied_Price', color='black', alpha=0.6, label='Price Index')
    
    # Color background based on Risk
    # We'll use axvspan for regions
    # But doing it for every day is slow. Let's do scatter points for risk.
    
    # Map risk to colors
    colors = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}
    risk_colors = [colors[r] for r in ticker_df['Predicted_Risk']]
    
    plt.scatter(ticker_df['Date'], ticker_df['Implied_Price'], c=risk_colors, s=10, alpha=0.8, label='Risk Level')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c, lw=4) for c in ['green', 'yellow', 'orange', 'red']]
    plt.legend(custom_lines, ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'], loc='upper left')
    
    plt.title(f'Price Trend vs Predicted Risk Level: {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Hypothetical Price Index (Start=100)')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'{OUTPUT_DIR}/risk_plot_{ticker}.png')
    print(f"Saved: {OUTPUT_DIR}/risk_plot_{ticker}.png")
    plt.close()

def main():
    rf, df = load_resources()
    
    # Must match training features
    features = [
        'Log_Return', 'RSI', 'Volat_20d', 'SMA_20_Ratio', 'Vol_Change', 'Rel_Return', 'Market_Corr',
        'SMA_50_Ratio', 'Return_90d', 'Volat_90d', 'Drawdown_90d', 'Bollinger_Pct', 'Bollinger_Width',
        'MARKET_Log_Return'
    ]
    
    plot_feature_importance(rf, features)
    plot_confusion_matrix(rf, df, features)
    
    # Examples
    plot_ticker_risk(df, 'NVDA', rf, features)
    plot_ticker_risk(df, 'TSLA', rf, features)
    plot_ticker_risk(df, 'AAPL', rf, features)

if __name__ == "__main__":
    main()
