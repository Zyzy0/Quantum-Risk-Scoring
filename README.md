# Quantum Risk Scoring System

## Project Overview
This project implements a Variational Quantum Classifier (VQC) to predict financial market risk levels. 
By leveraging Quantum Machine Learning (QML), the pipeline transforms high-dimensional classical market data into quantum states to classify future realized volatility into four distinct tiers: Low, Medium, High, and Very High.
## Project Structure
The repository is organized into a modular pipeline to ensure reproducibility:
* **data/**: Contains raw historical data and engineered feature sets.
* **src/**: Core logic including data preparation, the quantum circuit definition, and classical benchmarks.
* **notebooks/**: Experimental environment and final results visualization.
* **run_experiments.py**: The main execution script for training and evaluation.

## The Pipeline: Step-by-Step
Phase 1: Data Acquisition & Feature Engineering
* **Source**: Automated download of the top 100 tickers and S&P 500 benchmark via yfinance.
* **Feature Set**: Generated 701 technical features, including Log Returns, RSI, 20-day Volatility, and Market Beta.
* **Preprocessing**: Applied StandardScaler to normalize features for high-fidelity quantum embedding.

Phase 2: Quantum-Classical Bridge (PCA)
* To accommodate the constraints of Near-term Intermediate Scale Quantum (NISQ) devices, Principal Component Analysis (PCA) was used to reduce 701 features down to 4 principal components.
* These components are mapped to rotation angles $[0, \pi]$ for quantum state preparation.

Phase 3: Quantum Model Architecture (VQC)
* **Framework**: Built using PennyLane.
* **Embedding**: AngleEmbedding maps classical data onto 4 qubits.
* **Ansatz**: StronglyEntanglingLayers are used to create complex quantum correlations between features.
* **Optimization**: A hybrid approach using the Adam Optimizer and Cross-Entropy Loss to adjust quantum gate weights.

## Results & Visualization
The model was tested against the early 2025 market environment.

Key Findings:
* **Risk Detection**: The model demonstrated a high sensitivity to market turbulence, correctly identifying "High" risk regimes during 2025 volatility spikes.
* **Performance**: Achieved stable classification, though it showed a conservative bias (classifying some "Very High" instances as "High").
* **Comparison**: Outperformed random guessing by establishing a clear correlation with the benchmark realized volatility trend.

## Installation & Usage
1. Clone the repository:
```Bash
git clone https://github.com/your-username/Quantum-Risk-Scoring.git
```
2. Install dependencies:
```Bash
pip install -r .\requirements.txt
```
3. Run the full pipeline:
* Generate data: 
```Bash
python data_processing.py
python data_scaling_labeling.py
```
* Train model:
```Bash
python run_experiments.py
```
* View Dashboard: Open notebooks/04_results.ipynb
