# Markowitz FX Portfolio Optimizer

Dynamic foreign exchange portfolio optimization using Modern Portfolio Theory (Markowitz, 1952) with regime detection and deep learning extensions.

## Project Overview

This project implements a multi-phase quantitative research framework for FX portfolio management:

- **Phase 1:** Markowitz mean-variance optimization on FX pairs
- **Phase 2:** Hidden Markov Model regime detection for dynamic reallocation  
- **Phase 3:** LSTM-based return forecasting integrated with portfolio optimization

## Technologies

- Python 3.11
- MetaTrader5 (data source)
- pandas, NumPy, SciPy
- scikit-learn
- matplotlib / seaborn

## FX Universe

EUR/USD · GBP/USD · USD/JPY · EUR/JPY

## Methodology

1. Download OHLCV data via MetaTrader5 API
2. Calculate log returns
3. Build covariance matrix and expected returns
4. Optimize portfolio weights (minimum variance + maximum Sharpe ratio)
5. Plot efficient frontier

## Status

🔨 In progress — Phase 1

## Author

Łukasz Wilewski  
[LinkedIn](https://www.linkedin.com/in/łukasz-wilewski/)  
Finance & Investment Graduate | Quantitative Finance | Python
