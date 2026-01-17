# Project Pipeline: Step-by-Step
## Phase 1: Data Acquisition

- Source: Yahoo Finance (via yfinance library).
- Assets: Top 100 market tickers + S&P 500 (^GSPC) as a market benchmark.
- Period: Last 5 years of daily OHLCV (Open, High, Low, Close, Volume) data.
- Caching: Implemented a check-sum system to skip downloads if raw data already exists locally.

## Phase 2: Feature Engineering

Calculated 701 technical and relative features to provide the model with market context:

- Log Returns: Logarithmic transformation of price changes for statistical stability.
- Momentum (RSI): Relative Strength Index to identify overbought/oversold conditions.
- Volatility: Rolling 20-day standard deviation to measure historical risk.
- Trend (SMA Ratio): Distance from the 20-day moving average to identify trend deviations.
- Market Context: Relative returns (Alpha) and 60-day correlation (Beta) against the S&P 500.

## Phase 3: Preprocessing & Labeling

- Scaling: Applied StandardScaler to normalize features (Mean=0, Std=1), ensuring features like RSI (0-100) and Log Returns (-0.05 to 0.05) are treated equally by the model.

- Risk Labeling: Created a 4-tier classification target based on Future Realized Volatility (21-day forward-looking window):

        0: Low Risk

        1: Medium Risk

        2: High Risk

        3: Very High Risk

- Quantile Balancing: Used quantiles to ensure an even distribution of classes (25% per class) to prevent model bias.

## Future ML Steps (TODO List)

    [ ] Chronological Train/Test Split: Divide data into training (e.g., 2021–2024) and testing (2025) sets. Crucial: Do not use random shuffle to avoid look-ahead bias.

    [ ] Feature Selection / Dimensionality Reduction: * Evaluate feature importance using a Random Forest/XGBoost

    [ ] Apply PCA (Principal Component Analysis) to reduce 701 features to 4–8 components for Quantum Circuit compatibility.

    [ ] Classical Baseline Model: Train an XGBoost or Random Forest Classifier to establish a benchmark for accuracy and F1-Score.

    [ ] Quantum Circuit Implementation: * Develop a Variational Quantum Classifier (VQC) using PennyLane.

    [ ]Implement Angle Embedding to map classical components to Qubits.

    [ ] Risk Scoring Dashboard: Visualize the predicted risk levels for the upcoming month for the entire portfolio.