import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def create_risk_labels(df, tickers, horizon=21):
    """
    Creates risk labels based on future realized volatility over the next 'horizon' days.
    """
    print(f"Creating labels for a {horizon}-day investment horizon...")
    
    # We will store the future volatility for each ticker here
    future_vols = pd.DataFrame(index=df.index)

    for ticker in tickers:
        # Calculate future volatility (std dev of returns for the NEXT month)
        # We shift(-horizon) to align today's features with next month's volatility
        log_return_col = f'{ticker}_Log_Return'
        if log_return_col in df.columns:
            future_vol = df[log_return_col].rolling(window=horizon).std().shift(-horizon)
            future_vols[f'{ticker}_Future_Vol'] = future_vol

    # For simplicity, we can calculate a median risk across all stocks or do it per-stock.
    # Most ML models for stocks are trained on "stacked" data (long format).
    # Let's create a combined risk metric to define what "High Risk" means in this market.
    all_future_vols = future_vols.values.flatten()
    all_future_vols = all_future_vols[~np.isnan(all_future_vols)] # remove NaNs
    
    # Define thresholds based on quantiles (25%, 50%, 75%)
    thresholds = np.quantile(all_future_vols, [0.25, 0.5, 0.75])
    
    print(f"Risk Thresholds (Volatility): Low < {thresholds[0]:.4f} < Medium < {thresholds[1]:.4f} < High < {thresholds[2]:.4f} < Very High")
    
    return future_vols, thresholds

def prepare_ml_ready_data():
    features_path = 'data/features_ml_data.csv'
    raw_path = 'data/historical_market_data.csv'
    output_path = 'data/final_ml_dataset.csv'

    if not os.path.exists(features_path):
        print("Features file not found. Run the previous script first.")
        return

    # 1. Load data
    df = pd.read_csv(features_path, index_col=0)
    # Extract original tickers from column names
    tickers = list(set([col.split('_')[0] for col in df.columns if '_' in col and col != 'MARKET_Log_Return']))

    # 2. Create Labels (The 'Y' variable)
    # Target: Risk level for the next 21 days
    future_vols, thresholds = create_risk_labels(df, tickers)
    
    # 3. Scaling Features (The 'X' variables)
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    # We don't scale the index (Date), we scale the values
    feature_cols = df.columns
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=feature_cols, index=df.index)

    # 4. Final Dataset Construction
    # Since predicting 100 different stocks at once is complex, 
    # usually we pick one target stock OR create a model that predicts "Market Risk".
    # Let's save the scaled features and the future volatility labels separately for now.
    
    # To keep it simple for your project: we add the future volatility of the FIRST ticker as a sample target
    # In a real scenario, you'd probably reshape this to (samples * stocks, features)
    target_ticker = tickers[0]
    df_scaled['TARGET_Future_Vol'] = future_vols[f'{target_ticker}_Future_Vol']
    
    # Assign Risk Categories
    def categorize_risk(vol):
        if pd.isna(vol): return np.nan
        if vol <= thresholds[0]: return 0 # Low
        if vol <= thresholds[1]: return 1 # Medium
        if vol <= thresholds[2]: return 2 # High
        return 3 # Very High

    df_scaled['TARGET_Risk_Level'] = df_scaled['TARGET_Future_Vol'].apply(categorize_risk)

    # Drop rows where we don't have a target (the last 21 days of the dataset)
    final_df = df_scaled.dropna(subset=['TARGET_Risk_Level'])

    final_df = final_df.round(4)

    final_df.to_csv(output_path)
    print(f"Final ML Dataset saved to {output_path} with {final_df.shape[0]} samples.")
    print(f"Target selected for demo: {target_ticker}")

if __name__ == "__main__":
    prepare_ml_ready_data()