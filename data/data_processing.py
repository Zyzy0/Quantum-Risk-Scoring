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

BENCHMARK_TICKER = "^GSPC"

def download_raw_data(tickers, period="5y"):
    all_to_download = tickers + [BENCHMARK_TICKER]
    print(f"Downloading data for {len(tickers)} companies + {BENCHMARK_TICKER}...")
    data = yf.download(all_to_download, period=period, interval="1d", auto_adjust=True)
    return data

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def data_preparing():
    raw_path = 'data/historical_market_data.csv'
    clean_path = 'data/features_ml_data.csv'
    
    if not os.path.exists(raw_path):
        print("Error: Raw data file not found!")
        return

    print("Starting Feature Engineering...")
    full_df = pd.read_csv(raw_path, header=[0, 1], index_col=0)
    
    market_close = full_df['Close'][BENCHMARK_TICKER]
    market_log_return = np.log(market_close / market_close.shift(1))
    
    all_features = []
    tickers = [t for t in full_df.columns.get_level_values(1).unique() if t != BENCHMARK_TICKER]

    for ticker in tickers:
        try:
            temp = pd.DataFrame()
            close = full_df['Close'][ticker]
            volume = full_df['Volume'][ticker]

            log_return = np.log(close / close.shift(1))
            temp[f'{ticker}_Log_Return'] = log_return
            
            # --- Technical Indicators (Classic) ---
            temp[f'{ticker}_RSI'] = calculate_rsi(close)
            temp[f'{ticker}_Volat_20d'] = log_return.rolling(window=20).std()
            temp[f'{ticker}_SMA_20_Ratio'] = close / close.rolling(window=20).mean()
            
            # --- Advanced / Quarterly Indicators ---
            # 1. Medium Term Trend (50 days ~ 2.5 months)
            temp[f'{ticker}_SMA_50_Ratio'] = close / close.rolling(window=50).mean()
            
            # 2. Quarterly Performance (90 days)
            # Log return sum over 60 trading days (~3 months)
            temp[f'{ticker}_Return_90d'] = log_return.rolling(window=60).sum()
            temp[f'{ticker}_Volat_90d'] = log_return.rolling(window=60).std()
            
            # 3. Max Drawdown (Quarterly) - How much did it drop from peak in last quarter?
            roll_max = close.rolling(window=60).max()
            temp[f'{ticker}_Drawdown_90d'] = (close / roll_max) - 1.0
            
            # 4. Bollinger Bands (20 days, 2 std dev)
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            
            # %B Indicator: Where is price relative to bands? (0 = lower, 1 = upper)
            temp[f'{ticker}_Bollinger_Pct'] = (close - lower_band) / (upper_band - lower_band + 1e-9)
            
            # Band Width (Volatility measure)
            temp[f'{ticker}_Bollinger_Width'] = (upper_band - lower_band) / sma_20

            temp[f'{ticker}_Vol_Change'] = volume.pct_change()
            temp[f'{ticker}_Rel_Return'] = log_return - market_log_return
            temp[f'{ticker}_Market_Corr'] = log_return.rolling(window=60).corr(market_log_return)

            all_features.append(temp)
        except:
            continue

    final_df = pd.concat(all_features, axis=1)
    final_df['MARKET_Log_Return'] = market_log_return

    # --- KLUCZOWA POPRAWKA CZYSZCZENIA ---
    # 1. Usuwamy wiersze z samego początku, gdzie MARKET_Log_Return to NaN
    final_df = final_df.dropna(subset=['MARKET_Log_Return'])
    
    # 2. Usuwamy dni, gdzie więcej niż 30% kolumn to NaN (czyli np. pierwsze 60 dni sesyjnych)
    # To sprawi, że w pliku nie będzie "pustych" dni na starcie.
    min_nonzero_cols = int(len(final_df.columns) * 0.7)
    final_df = final_df.dropna(thresh=min_nonzero_cols)
    
    # 3. Resztę braków (np. pojedyncze braki w Volume) uzupełniamy
    final_df = final_df.ffill().fillna(0)
    
    final_df = final_df.round(4)
    final_df.to_csv(clean_path)
    
    print(f"Success! Features saved: {clean_path}")
    print(f"Final shape: {final_df.shape}")

def main():
    if not os.path.exists('data'): os.makedirs('data')
    raw_data_file = "data/historical_market_data.csv"

    if not os.path.exists(raw_data_file):
        df = download_raw_data(top_100_tickers)
        df.to_csv(raw_data_file)
    
    data_preparing()

if __name__ == "__main__":
    main()