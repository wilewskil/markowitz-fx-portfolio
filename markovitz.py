import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize  


def calculate_expected_returns(log_returns):
    return log_returns.mean() * 1512 # Annualize the expected returns

def calculate_covariance_matrix(log_returns):
    return log_returns.cov() * 1512  # Annualize the covariance matrix

def portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.04):
    ret, vol = portfolio_performance(weights, expected_returns, cov_matrix)
    return (ret - risk_free_rate) / vol

def minimum_variance_portfolio(expected_returns, cov_matrix):
    n_assets = len(expected_returns)
    initial_weights = np.array([1/n_assets] * n_assets)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum of weights = 1
    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(
        fun=lambda w: portfolio_performance(w, expected_returns, cov_matrix)[1],  # Minimize volatility
        x0=initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result
def maximum_sharpe_portfolio(expected_returns, cov_matrix):
    n_assets = len(expected_returns)
    initial_weights = np.array([1/n_assets] * n_assets)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum of weights = 1
    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(
        fun=lambda w: -sharpe_ratio(w, expected_returns, cov_matrix),  # Maximize Sharpe ratio
        x0=initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

def generate_random_portfolios(n_portfolios, expected_returns, cov_matrix):
    returns = []
    volatilities = []
    sharpe_ratios = []
    weights_list = []
    for _ in range(n_portfolios):
        n_assets = len(expected_returns)
        w = np.random.random(n_assets)
        w = w / np.sum(w)
        ret, vol = portfolio_performance(w, expected_returns, cov_matrix)
        sharpe = sharpe_ratio(w, expected_returns, cov_matrix)

        returns.append(ret)
        volatilities.append(vol)
        sharpe_ratios.append(sharpe)
        weights_list.append(w)

    return {
        'returns': np.array(returns),
        'volatilities': np.array(volatilities),
        'sharpe_ratios': np.array(sharpe_ratios),
        'weights': np.array(weights_list)
        }

if __name__ == "__main__":
    from data_loader import get_all_pairs, calculate_log_returns
    import MetaTrader5 as mt5
    
    symbols = ['EURUSD', 'GBPUSD', 'EURJPY', 'USDJPY']
    timeframe = mt5.TIMEFRAME_H4
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    # Get data
    df = get_all_pairs(symbols, timeframe, start_date, end_date)
    log_returns = calculate_log_returns(df)
    
    # Calculate inputs
    expected_returns = calculate_expected_returns(log_returns)
    cov_matrix = calculate_covariance_matrix(log_returns)
    
    # Optimize
    min_var = minimum_variance_portfolio(expected_returns, cov_matrix)
    max_sharpe = maximum_sharpe_portfolio(expected_returns, cov_matrix)
    
    print("Minimum Variance Portfolio weights:")
    print(dict(zip(symbols, min_var.x.round(4))))
    
    print("\nMaximum Sharpe Portfolio weights:")
    print(dict(zip(symbols, max_sharpe.x.round(4))))

    results = generate_random_portfolios(5000, expected_returns, cov_matrix)
    print(f"Generated {len(results['returns'])} portfolios")
    print(f"Return range: {results['returns'].min():.4f} to {results['returns'].max():.4f}")
    print(f"Volatility range: {results['volatilities'].min():.4f} to {results['volatilities'].max():.4f}")