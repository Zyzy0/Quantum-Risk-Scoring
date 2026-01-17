import yfinance as yf
import pandas as pd
import os

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

def download_portfolio_data(tickers, period="5y"):
    print(f"Downloading data for {len(tickers)} companies...")
    
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True)

    if 'Close' in data.columns.levels[0]:
        prices = data['Close']
    else:
        prices = data

    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.9))
    prices = prices.ffill()

    return prices

def data_preparing():
    raw_path = 'data/historical_market_data.csv'
    clean_path = 'data/historical_market_data_cleaned.csv'
    
    if os.path.exists(raw_path):
        print("Cleaning data...")
        df = pd.read_csv(raw_path, index_col='Date')
        df = df.round(4)
        df = df.dropna(how='all')

        df.to_csv(clean_path)
        print(f"Data prepared and saved to {clean_path}")
    else:
        print("Error: Raw data file not found. Cannot prepare data.")

def main():
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Folder 'data' created.")

    raw_data_file = "data/historical_market_data.csv"

    if os.path.exists(raw_data_file):
        print(f"File '{raw_data_file}' already exists. Skipping download.")
    else:
        df = download_portfolio_data(top_100_tickers)
        df.to_csv(raw_data_file)
        print(f"Download complete. Data saved in {raw_data_file}")

    data_preparing()

if __name__ == "__main__":
    main()