import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def create_risk_labels(df, tickers, horizon=21):
    """
    Creates risk labels based on future realized volatility over the next 'horizon' days.
    """
    print(f"Creating labels for a {horizon}-day investment horizon...")
    
    future_vols_list = []

    # Iterate over all tickers to calculate future volatility
    for ticker in tickers:
        log_return_col = f'{ticker}_Log_Return'
        if log_return_col in df.columns:
            # Shift backwards to align today's row with NEXT month's volatility
            future_vol = df[log_return_col].rolling(window=horizon).std().shift(-horizon)
            
            # Store temporarily
            temp_df = pd.DataFrame(index=df.index)
            temp_df['Future_Vol'] = future_vol
            temp_df['Ticker'] = ticker
            future_vols_list.append(temp_df)

    # Concatenate all (this is just for threshold calculation globally)
    # We want global thresholds to ensure "High Risk" means the same thing for AAPL as for TSLA
    all_future_vols = pd.concat(future_vols_list)['Future_Vol']
    all_future_vols = all_future_vols.dropna()
    
    # Define thresholds based on quantiles (25%, 50%, 75%)
    thresholds = np.quantile(all_future_vols, [0.25, 0.5, 0.75])
    
    print(f"Global Risk Thresholds (Volatility): Low < {thresholds[0]:.4f} < Medium < {thresholds[1]:.4f} < High < {thresholds[2]:.4f} < Very High")
    
    return thresholds

def prepare_ml_ready_data():
    features_path = 'data/features_ml_data.csv'
    output_path = 'data/final_ml_dataset.csv'

    if not os.path.exists(features_path):
        print("Features file not found. Run the previous script first.")
        return

    # 1. Load data
    print("Loading feature data...")
    df = pd.read_csv(features_path, index_col=0)
    
    # Extract original tickers by looking at columns ending in '_Log_Return'
    # (filtering out MARKET which is special)
    tickers = list(set([col.split('_')[0] for col in df.columns if '_Log_Return' in col and 'MARKET' not in col]))
    print(f"Found {len(tickers)} tickers to process.")

    # 2. Establish Global Risk Thresholds
    thresholds = create_risk_labels(df, tickers)

    # 3. Transform to Long Format (Stacked)
    # We want a DataFrame with columns: [Date, Ticker, RSI, Log_Return, Volat_20d, ..., MARKET_Log_Return, TARGET]
    print("re-shaping data to Long Format (this may take a moment)...")
    
    master_data = []

    # Common features that apply to all stocks (Market data)
    market_col = 'MARKET_Log_Return'
    market_data = df[market_col]

    for ticker in tickers:
        # Identify columns for this specific ticker
        # Expected: {ticker}_Log_Return, {ticker}_RSI, {ticker}_Volat_20d, etc.
        # We rename them to generic names: Log_Return, RSI, Volat_20d...
        
        # Helper to safely get series
        def get_col(suffix):
            col_name = f"{ticker}_{suffix}"
            return df[col_name] if col_name in df.columns else None

        # Prepare a slice for this ticker
        ticker_df = pd.DataFrame(index=df.index)
        ticker_df['Date'] = df.index
        ticker_df['Ticker'] = ticker
        
        # Features
        ticker_df['Log_Return'] = get_col('Log_Return')
        ticker_df['RSI'] = get_col('RSI')
        ticker_df['Volat_20d'] = get_col('Volat_20d')
        ticker_df['SMA_20_Ratio'] = get_col('SMA_20_Ratio')
        ticker_df['Vol_Change'] = get_col('Vol_Change')
        ticker_df['Rel_Return'] = get_col('Rel_Return')
        ticker_df['Market_Corr'] = get_col('Market_Corr')
        
        # --- New Advanced Features ---
        ticker_df['SMA_50_Ratio'] = get_col('SMA_50_Ratio')
        ticker_df['Return_90d'] = get_col('Return_90d')
        ticker_df['Volat_90d'] = get_col('Volat_90d')
        ticker_df['Drawdown_90d'] = get_col('Drawdown_90d')
        ticker_df['Bollinger_Pct'] = get_col('Bollinger_Pct')
        ticker_df['Bollinger_Width'] = get_col('Bollinger_Width')
        
        # Add Market Context
        ticker_df['MARKET_Log_Return'] = market_data
        
        # Create Target (Future Volatility)
        # Shift -21 days
        ticker_df['TARGET_Future_Vol'] = ticker_df['Log_Return'].rolling(window=21).std().shift(-21)
        
        master_data.append(ticker_df)

    # Combine all
    full_df = pd.concat(master_data, ignore_index=True)
    
    # 4. Clean & Scale
    print(f"Combined data shape before cleaning: {full_df.shape}")
    
    # Remove rows where Target is NaN (the last 21 days for each stock) or features are missing
    full_df.dropna(inplace=True)
    
    print(f"Shape after dropping NaNs: {full_df.shape}")

    # Scale specific Feature columns
    # We do NOT scale 'Date', 'Ticker', 'TARGET_Future_Vol'
    feature_cols = [
        'Log_Return', 'RSI', 'Volat_20d', 'SMA_20_Ratio', 'Vol_Change', 'Rel_Return', 'Market_Corr',
        'SMA_50_Ratio', 'Return_90d', 'Volat_90d', 'Drawdown_90d', 'Bollinger_Pct', 'Bollinger_Width',
        'MARKET_Log_Return'
    ]
    
    print("Scaling features...")
    scaler = StandardScaler()
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])

    # 5. Labeling
    def categorize_risk(vol):
        if vol <= thresholds[0]: return 0 # Low
        if vol <= thresholds[1]: return 1 # Medium
        if vol <= thresholds[2]: return 2 # High
        return 3 # Very High

    full_df['TARGET_Risk_Level'] = full_df['TARGET_Future_Vol'].apply(categorize_risk)

    # Save
    full_df = full_df.round(4)
    full_df.to_csv(output_path, index=False)
    
    print(f"Final Global ML Dataset saved to {output_path}")
    print(f"Total Samples: {len(full_df)}")
    print("Columns:", full_df.columns.tolist())

if __name__ == "__main__":
    prepare_ml_ready_data()