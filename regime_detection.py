import numpy as np 
import pandas as pd 
from hmmlearn import hmm
import matplotlib.pyplot as plt


def detect_regimes(log_returns, n_states=3):
    clean_returns = log_returns.dropna()
    model = hmm.GaussianHMM(n_components=n_states, n_iter=1000, random_state=42)
    model.fit(clean_returns)
    states = model.predict(clean_returns)
    return states, model

def get_regime_portfolios(log_returns, states, n_states):
    from markowitz import calculate_expected_returns, calculate_covariance_matrix
    from markowitz import minimum_variance_portfolio, maximum_sharpe_portfolio

    regime_portfolios = {}

    for regime in range(n_states):
        regime_returns = log_returns[states == regime]
        expected_returns = calculate_expected_returns(regime_returns)
        cov_matrix = calculate_covariance_matrix(regime_returns)

        regime_portfolios[regime] = {
            'min_var': minimum_variance_portfolio(expected_returns, cov_matrix),
            'max_sharpe': maximum_sharpe_portfolio(expected_returns, cov_matrix),
            'expected_returns': expected_returns,
            'cov_matrix': cov_matrix
        }

    return regime_portfolios


if __name__ == "__main__":
    from data_loader import get_all_pairs, calculate_log_returns
    import MetaTrader5 as mt5
    import numpy as np
    from visualizer import plot_regimes

    symbols = ['EURUSD', 'GBPUSD', 'EURJPY', 'USDJPY']
    timeframe = mt5.TIMEFRAME_H4
    start_date = "2020-01-01"
    end_date = "2024-01-01"
        
        # Get data
    df = get_all_pairs(symbols, timeframe, start_date, end_date)
    log_returns = calculate_log_returns(df)

    states, model = detect_regimes(log_returns)
    print("Detected regimes:")
    print(states)


    unique, counts = np.unique(states, return_counts=True)
    for regime, count in zip(unique, counts):
        print(f"Regime {regime}: {count} periods ({count/len(states)*100:.1f}%)")
    clean_returns = log_returns.dropna()

    for regime in range(3):
        mask = states == regime
        regime_returns = clean_returns[mask]
        print(f"\nRegime {regime}:")
        print(f"  Mean return: {regime_returns.mean().mean():.6f}")
        print(f"  Volatility: {regime_returns.std().mean():.6f}")

    regime_portfolios = get_regime_portfolios(log_returns.dropna(), states, 3)
    for regime, portfolio in regime_portfolios.items():
        print(f"\nRegime {regime}:")
        print(f"  Min Variance weights: {portfolio['min_var'].x.round(3)}")
        print(f"  Max Sharpe weights: {portfolio['max_sharpe'].x.round(3)}")

    plot_regimes(df.iloc[1:], states, symbols)

    
