import numpy as np
import pandas as pd
import vectorbt as vbt
import argparse

# Function to run backtest based on custom thresholds
def run_backtest(entry_threshold, exit_threshold, spread_preds, dates, hedge_ratios, price_df, optimize):
    # Define the entry/exit signals based on custom thresholds
    long_short_entry = spread_preds < -entry_threshold  # Element-wise comparison
    long_short_exit = spread_preds >= exit_threshold  # Element-wise comparison
    
    short_long_entry = spread_preds > entry_threshold  # Element-wise comparison
    short_long_exit = spread_preds <= -exit_threshold  # Element-wise comparison

    # Define the position sizes for sym_1 and sym_2 (positive, with hedge ratio applied)
    size_sym_1 = np.ones(len(dates))  # Fixed size of 1 for sym_1
    size_sym_2 = hedge_ratios  # Hedge ratio size for sym_2

    # Create size matrix as a DataFrame (to align with signals and prices)
    size_matrix = pd.DataFrame({
        'sym_1': size_sym_1,
        'sym_2': size_sym_2
    }, index=dates)

    # Combine the entry signals for both assets (sym_1 and sym_2 must be handled separately)
    entries = pd.DataFrame({
        'sym_1': long_short_entry,  # Long sym_1
        'sym_2': short_long_entry   # Short sym_2
    }, index=dates)

    exits = pd.DataFrame({
        'sym_1': long_short_exit,   # Exit long sym_1
        'sym_2': short_long_exit    # Exit short sym_2
    }, index=dates)

    # Similarly handle short entries and exits for opposite positions
    short_entries = pd.DataFrame({
        'sym_1': short_long_entry,  # Short sym_1
        'sym_2': long_short_entry   # Long sym_2
    }, index=dates)

    short_exits = pd.DataFrame({
        'sym_1': short_long_exit,   # Exit short sym_1
        'sym_2': long_short_exit    # Exit long sym_2
    }, index=dates)

    # Run backtest using vectorbt's from_signals method
    portfolio = vbt.Portfolio.from_signals(
        price_df,  # Price data for sym_1 and sym_2
        entries=entries,  # Entries for both sym_1 and sym_2
        exits=exits,  # Exits for both sym_1 and sym_2
        short_entries=short_entries,  # Short entries for both sym_1 and sym_2
        short_exits=short_exits,  # Short exits for both sym_1 and sym_2
        size=size_matrix,  # Size matrix for sym_1 and sym_2
        freq='1D',
        fees=0.001,
        slippage=0.001
    )

    # Return the Sharpe ratio as a scalar by taking the mean of the two assets
    if optimize:
        return portfolio.sharpe_ratio().mean()  # Take the mean Sharpe ratio across both assets
    else:
        stats = portfolio.stats()
        return stats

# Function to perform grid search optimization
def optimize_thresholds(spread_preds, dates, hedge_ratios, price_df, optimize):
    # Define a range of entry and exit thresholds to test
    entry_thresholds = np.linspace(0, 3.0, 50)
    exit_thresholds = np.linspace(0, 3.0, 50)

    # Perform grid search optimization
    results = {}
    for entry_threshold in entry_thresholds:
        for exit_threshold in exit_thresholds:
            sharpe_ratio = run_backtest(entry_threshold, exit_threshold, spread_preds, dates, hedge_ratios, price_df, optimize)
            results[(entry_threshold, exit_threshold)] = sharpe_ratio

    # Find the best combination of entry/exit thresholds
    best_thresholds = max(results, key=results.get)  # No ambiguity since Sharpe ratio is scalar now
    best_sharpe = results[best_thresholds]
    #print(f"Best entry/exit thresholds: {best_thresholds}, Best Sharpe ratio: {best_sharpe}")

    return best_thresholds, best_sharpe

def main(df, optimize=False, entry_threshold=2, exit_threshold=0.5):
    df = pd.read_csv(df)
    df.set_index('date', inplace=True)
    dates = df.index
    sym_1_prices = df.adj_close_sym_1.values
    sym_2_prices = df.adj_close_sym_2.values

    # Create a price dataframe for sym_1 and sym_2
    price_df = pd.DataFrame({
        'sym_1_price': sym_1_prices,
        'sym_2_price': sym_2_prices
    }, index=dates)

    spread_preds = df.spread_zscore
    hedge_ratios = df.hedge_ratio.to_numpy()
    print(optimize)

    if optimize:
        # Perform optimization to find the best entry/exit thresholds
        best_thresholds, best_sharpe = optimize_thresholds(spread_preds, dates, hedge_ratios, price_df, optimize)
        print(f"best_thresholds: {best_thresholds}")
        print(f"best_sharpe: {best_sharpe}")
    else:
        # Run a single backtest with predefined entry/exit thresholds
        stats = run_backtest(entry_threshold, exit_threshold, spread_preds, dates, hedge_ratios, price_df, optimize)
        stats = stats.drop(['Start','End'])
        stats.to_csv('traditional_strategy_perf.csv')

        #sharpe = run_backtest(entry_threshold=1.0, exit_threshold=0.5)
        #print(f"Sharpe ratio for predefined thresholds: {sharpe}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get pairs for selected index and date range")
    parser.add_argument('df', type=str, help='Formed strategy data')
    parser.add_argument('optimize', type=str, help='Run entry/exit optimisation')
    parser.add_argument('entry_threshold', type=str, help='Entry threshold zsore')
    parser.add_argument('exit_threshold', type=str, help='Exit threshold zsore')

    # Parse arguments
    args = parser.parse_args()
    df = args.df
    optimize = eval(args.optimize)
    entry_threshold = float(args.entry_threshold)
    exit_threshold = float(args.exit_threshold)
    
    main(df, optimize, entry_threshold, exit_threshold)
    
    """
    # Call the main function with parsed arguments
    df = 'traditional_strategy.csv'
    optimize = True
    entry_threshold = 1.0
    exit_threshold = 0.0
    main(df, optimize, entry_threshold, exit_threshold)
    """