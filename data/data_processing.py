import yfinance as yf
import pandas as pd
import numpy as np
import os

# List of target tickers
top_100_tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "TSM", "LLY",
    "AVGO", "V", "JPM", "WMT", "UNH", "MA", "XOM", "ORCL", "ASML", "COST",
    "PG", "HD", "JNJ", "ABBV", "MRK", "BAC", "CVX", "CRM", "NFLX", "AMD",
    "ADBE", "TMO", "LIN", "QCOM", "PDD", "SAP", "TM", "WFC", "INTU", "ACN",
    "IBM", "AMGN", "CAT", "TXN", "UBER", "GE", "ISRG", "PFE", "NOW", "AMAT",
    "PM", "LOW", "VRTX", "HON", "UNP", "BKNG", "RY", "RTX", "SYK", "MS",
    "SPGI", "HSBC", "GS", "COP", "REGN", "SCHW", "ETN", "TJX", "PLTR", "LMT",
    "PGR", "CB", "BSX", "PANW", "VZ", "MU", "SHEL", "TTE", "RYAN", "LRCX",
    "BLK", "ADI", "KLAC", "MDLZ", "AMT", "FIS", "HCA", "BA", "DE", "ITW",
    "SNY", "CDNS", "BP", "BX", "EQIX", "T", "ABNB", "C", "MCD", "NKE"
]

BENCHMARK_TICKER = "^GSPC"  # S&P 500 Index

def download_raw_data(tickers, period="5y"):
    # Add S&P 500 to the download list for comparison
    all_to_download = tickers + [BENCHMARK_TICKER]
    print(f"Downloading OHLCV data for {len(tickers)} companies + benchmark {BENCHMARK_TICKER}...")
    data = yf.download(all_to_download, period=period, interval="1d", auto_adjust=True)
    return data

def calculate_rsi(series, period=14):
    """Calculate the Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)  # Add small epsilon to avoid division by zero
    return 100 - (100 / (1 + rs))

def data_preparing():
    raw_path = 'data/historical_market_data.csv'
    clean_path = 'data/features_ml_data.csv'
    
    if not os.path.exists(raw_path):
        print("Error: Raw data file not found!")
        return

    print("Starting Feature Engineering with S&P 500 comparison...")
    # Load multi-index dataframe (yfinance format)
    full_df = pd.read_csv(raw_path, header=[0, 1], index_col=0)
    
    # 1. Extract Benchmark (S&P 500) data
    market_close = full_df['Close'][BENCHMARK_TICKER]
    market_log_return = np.log(market_close / market_close.shift(1))
    
    all_features = []
    # Get tickers excluding the benchmark itself
    tickers = [t for t in full_df.columns.get_level_values(1).unique() if t != BENCHMARK_TICKER]

    for ticker in tickers:
        try:
            temp = pd.DataFrame()
            close = full_df['Close'][ticker]
            volume = full_df['Volume'][ticker]

            # --- INDIVIDUAL TECHNICAL FEATURES ---
            # Log Returns: More stable for ML models than simple percentages
            log_return = np.log(close / close.shift(1))
            temp[f'{ticker}_Log_Return'] = log_return
            
            # RSI: Relative Strength Index (Momentum indicator)
            temp[f'{ticker}_RSI'] = calculate_rsi(close)
            
            # Volatility: Rolling 20-day standard deviation of log returns
            temp[f'{ticker}_Volat_20d'] = log_return.rolling(window=20).std()
            
            # SMA Ratio: Current Price / 20-day Moving Average (Trend deviation)
            temp[f'{ticker}_SMA_20_Ratio'] = close / close.rolling(window=20).mean()
            
            # Volume Change: Percentage change in daily trading volume
            temp[f'{ticker}_Vol_Change'] = volume.pct_change()

            # --- MARKET COMPARISON FEATURES (Alpha & Beta) ---
            # Relative Return: Outperformance/Underperformance vs S&P 500
            temp[f'{ticker}_Rel_Return'] = log_return - market_log_return
            
            # Rolling Correlation: How closely the stock tracks the market (60-day window)
            temp[f'{ticker}_Market_Corr'] = log_return.rolling(window=60).corr(market_log_return)

            all_features.append(temp)
        except Exception as e:
            print(f"Skipped {ticker} due to error: {e}")

    # Combine all features into one master dataframe
    final_df = pd.concat(all_features, axis=1)

    final_df = final_df.copy()
    
    # Add market returns as a standalone feature
    final_df['MARKET_Log_Return'] = market_log_return
    
    # Final cleanup: Round decimals and drop initial NaN rows created by rolling windows
    final_df = final_df.round(4).dropna()
    
    final_df.to_csv(clean_path)
    print(f"Success! ML-ready features saved to: {clean_path}")
    print(f"Total features generated: {final_df.shape[1]}")

def main():
    if not os.path.exists('data'):
        os.makedirs('data')

    raw_data_file = "data/historical_market_data.csv"

    # Check for cache
    if os.path.exists(raw_data_file):
        print("Raw data already exists. Proceeding to Feature Engineering...")
    else:
        df = download_raw_data(top_100_tickers)
        df.to_csv(raw_data_file)
        print("Raw data download complete.")

    # Always run the preparation to ensure features are up to date
    data_preparing()

if __name__ == "__main__":
    main()