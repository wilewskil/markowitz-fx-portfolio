import matplotlib.pyplot as plt
from markowitz import portfolio_performance 

def plot_efficient_frontier(results, min_var, max_sharpe, expected_returns, cov_matrix, symbols):
    plt.figure(figsize=(10, 6))
    plt.scatter(results['volatilities'], results['returns'], c=results['sharpe_ratios'], cmap='viridis', alpha=0.5)
        # Minimum Variance Portfolio
    min_var_ret, min_var_vol = portfolio_performance(min_var.x, expected_returns, cov_matrix)
    plt.scatter(min_var_vol, min_var_ret, color='red', marker='*', s=300, label='Min Variance')

    # Maximum Sharpe Portfolio  
    max_sharpe_ret, max_sharpe_vol = portfolio_performance(max_sharpe.x, expected_returns, cov_matrix)
    plt.scatter(max_sharpe_vol, max_sharpe_ret, color='green', marker='*', s=300, label='Max Sharpe')

    plt.legend()
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.show()

