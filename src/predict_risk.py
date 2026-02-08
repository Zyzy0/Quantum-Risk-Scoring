import pandas as pd
import numpy as np
import joblib
import os
import sys

# Configuration
DATA_PATH = 'data/final_ml_dataset.csv'
MODEL_PATH = 'models/rf_global_model.pkl'

# Feature names used during training (MUST MATCH EXACTLY)
FEATURES = [
    'Log_Return', 'RSI', 'Volat_20d', 'SMA_20_Ratio', 'Vol_Change', 'Rel_Return', 'Market_Corr',
    'SMA_50_Ratio', 'Return_90d', 'Volat_90d', 'Drawdown_90d', 'Bollinger_Pct', 'Bollinger_Width',
    'MARKET_Log_Return'
]

def load_resources():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}.")
        print("Please run 'src/random_forest_model.py' to train the model first.")
        sys.exit(1)
        
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}.")
        sys.exit(1)

    print("Loading model and data...")
    rf = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return rf, df

def predict_risk(ticker, rf, df):
    ticker = ticker.upper()
    ticker_df = df[df['Ticker'] == ticker].copy()
    
    if ticker_df.empty:
        print(f"❌ No data found for ticker: {ticker}")
        print("Available tickers (first 20):", df['Ticker'].unique()[:20])
        return

    # Sort by date
    ticker_df = ticker_df.sort_values(by='Date')
    
    # Get latest available date
    latest_row = ticker_df.iloc[-1]
    latest_date = latest_row['Date']
    
    # Get features for prediction
    X_new = ticker_df.iloc[[-1]][FEATURES]
    
    # Prediction
    prediction = rf.predict(X_new)[0]
    probabilities = rf.predict_proba(X_new)[0]
    
    # Map classes
    risk_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    predicted_label = risk_labels.get(prediction, "Unknown")
    
    # Display Result
    print(f"\n===== RISK ASSESSMENT: {ticker} =====")
    print(f"📅 Date: {latest_date.date()}")
    print(f"📊 Predicted Risk Level: {predicted_label}")
    
    print("\nConfidence Matrix:")
    for i, label in risk_labels.items():
        if i < len(probabilities):
            # Simple bar chart
            prob = probabilities[i]
            bar = "█" * int(prob * 20)
            print(f"  {label:<12} {prob:.1%} {bar}")
            
    # Key Factors (Why?)
    # We show the values of the top 3 most important features for this specific prediction
    print("\n🔍 Key Indicators (Current vs Average):")
    
    # Calculate simple stats for context
    # (In a real app, use SHAP values, but simple feature values work for now)
    avg_volat_90 = df['Volat_90d'].mean()
    curr_volat_90 = latest_row['Volat_90d']
    
    avg_drawdown = df['Drawdown_90d'].mean()
    curr_drawdown = latest_row['Drawdown_90d']
    
    print(f"  • Quarterly Volatility: {curr_volat_90:.4f} (Avg: {avg_volat_90:.4f})")
    if curr_volat_90 > avg_volat_90:
        print("    -> Higher than average volatility increases risk.")
    else:
        print("    -> Lower than average volatility suggests stability.")
        
    print(f"  • Max Drawdown (90d):   {curr_drawdown:.1%} (Avg: {avg_drawdown:.1%})")
    print(f"  • Bollinger Width:      {latest_row['Bollinger_Width']:.4f}")

def main():
    rf, df = load_resources()
    
    while True:
        print("\n" + "-"*40)
        user_input = input("Enter Ticker Symbol (or 'q' to quit): ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
            
        if not user_input:
            continue
            
        predict_risk(user_input, rf, df)

if __name__ == "__main__":
    main()
