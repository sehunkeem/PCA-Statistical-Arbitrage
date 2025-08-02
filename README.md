# Implementation of Avellaneda & Lee’s PCA Statistical Arbitrage Strategy

The repository contains an implementation for the paper,
*“Statistical Arbitrage in the U.S. Equities Market”* by M. Avellaneda and J. Lee (2008).

It contains a full pipeline - from raw data ingestion to signal generation, hedge construction, portfolio execution, and performance evaluation - based on the PCA statistical arbitrage methodology.

## Contents

- **Data ingestion:** `yf_download.py` downloads daily OHLCV data from Yahoo Finance, structures it in a local data lake (bronze/silver/gold), and constructs wide-form price and market-cap dataframes.
- **Strategy core:** `main.py` orchestrates the backtest using the processed data.
- **Configuration:** All model and execution hyperparameters live in the `ALConfig` dataclass in `algorithm.py`.
- **Notebook walkthroughs:**  
  - `experiment.ipynb` — implementation details, hyperparameter experiments, and backtest results.  
  - `detailed_summary.ipynb` — summary of the original Avellaneda-Lee paper, as well as derivations that are omitted in the paper.
