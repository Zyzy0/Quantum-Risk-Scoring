# 📊 Data Pipeline & Machine Learning Methodology

This directory handles the end-to-end data processing for the **Quantum Risk Scoring** project. The pipeline transforms raw historical market data into a machine-learning-ready dataset used to predict investment risk.

## 🔄 Pipeline Overview

1.  **Data Collection** (`data_processing.py`):
    *   Downloads 5 years of daily history for ~100 top companies (Apple, Microsoft, Tesla, etc.) + S&P 500 Index.
    *   Source: Yahoo Finance via `yfinance`.
    *   Output: `historical_market_data.csv` (Raw OHLCV data).

2.  **Feature Engineering** (`data_processing.py`):
    *   Calculates technical indicators focusing on **Medium-Term (Quarterly) Trends**:
        *   **`Volat_90d`**: 3-month Rolling Volatility (Crucial for risk).
        *   **`Return_90d`**: 3-month Cumulative Return.
        *   **`Drawdown_90d`**: Maximum loss from peak in the last quarter (Crash detection).
        *   **`SMA_50_Ratio`**: Price relative to 50-day Moving Average (Trend strength).
        *   **`Bollinger_Width`**: Volatility compression indicator.
    *   Output: `features_ml_data.csv` (Wide format).

3.  **Data Transformation & Labeling** (`data_scaling_labeling.py`):
    *   **Reshaping**: Converts "Wide" data (one row per day, columns for all tickers) to "Long" format (one row per Ticker-Day).
    *   **Scaling**: Standardizes features using `StandardScaler` (Mean=0, Std=1).
    *   **Labeling (The Target)**:
        *   We predict **Future Volatility** over the NEXT 21 days (1 month).
        *   **Classes**:
            *   🟢 **Low Risk**: Bottom 25% of volatility.
            *   🟡 **Medium Risk**: 25-50% percentile.
            *   🟠 **High Risk**: 50-75% percentile.
            *   🔴 **Very High Risk**: Top 25% (Extreme volatility).
    *   Output: `final_ml_dataset.csv` (Ready for training).

## 🧠 Model Methodology (`src/random_forest_model.py`)

We use a **Global Random Forest Classifier** trained on all companies simultaneously.

*   **Algorithm**: Random Forest (Ensemble of Decision Trees).
*   **Validation Strategy**: **Temporal Split**.
    *   We train on data from 2020-2023.
    *   We test on the "future" (2024-2025).
    *   *Why?* Standard random splitting would cause "data leakage" in financial time series.
*   **Hyperparameter Tuning**:
    *   Model parameters were optimized using `GridSearchCV`.
    *   **Class Balancing**: Weights are adjusted to pay more attention to "High Risk" events.

## 📈 Key Findings

The model identified that **Quarterly Volatility (`Volat_90d`)** is the single most important predictor of future risk (41% importance), far outweighing short-term daily fluctuations. This confirms that for medium-term investing, looking at the 3-month trend is more valuable than yesterday's price change.