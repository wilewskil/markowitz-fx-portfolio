import matplotlib.pyplot as plt
from markowitz import portfolio_performance 
import numpy as np
from matplotlib.patches import Rectangle

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

def plot_regimes(df, states, symbols):
    colors = {0: 'lightblue', 1: 'lightgreen', 2: 'salmon'}
    labels = {0: 'Normal', 1: 'Low Volatility', 2: 'Crisis'}
    
    plt.figure(figsize=(14, 10))
    for i, symbol in enumerate(symbols):
        plt.subplot(len(symbols), 1, i + 1)
        plt.plot(df.index, df[symbol], color='black', linewidth=0.8)
        plt.title(f"{symbol} Price with Regimes")
        plt.ylabel("Price")
        
        # Color background by regime
        for j in range(len(states)):
            plt.axvspan(df.index[j], df.index[min(j+1, len(df.index)-1)],
                       alpha=0.3, color=colors[states[j]], label=labels[states[j]])
        
        # Clean legend - no duplicates
        handles = [plt.Rectangle((0,0),1,1, color=colors[r], alpha=0.3) 
                   for r in colors]
        plt.legend(handles, labels.values(), loc='upper right')
    
    plt.tight_layout()
    plt.show()