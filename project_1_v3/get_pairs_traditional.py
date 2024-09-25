import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import itertools
from sklearn.linear_model import LinearRegression
import data_manager as dm
import multiprocessing as mp
import warnings
import random
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import coint
from tqdm import tqdm
import argparse
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller
import random
random.seed(42)
np.random.seed(42)


cwd = os.getcwd()
if 'project' in cwd:
    conn = sqlite3.connect('..\\equities_data.db')
else:    
    # Connect to the SQLite database
    conn = sqlite3.connect('equities_data.db')

# Connect to the SQLite database
#conn = sqlite3.connect('equities_data.db')
    
def calculate_cointegration_strength(asset1_prices, asset2_prices):
    # Perform Engle-Granger cointegration test
    score, p_value, _ = coint(asset1_prices, asset2_prices)
    
    # Return the p-value as the "cointegration strength" (lower is stronger)
    return p_value

def estimate_hedge_ratio(asset1_prices, asset2_prices):
    # Kalman filter for dynamic hedge ratio
    # Define the observation matrix and the transition matrix
    observation_matrix = np.vstack([asset2_prices, np.ones(len(asset2_prices))]).T.reshape(-1, 1, 2)  # [sym_2_price, intercept]
    transition_matrix = np.eye(2)  # Hedge ratio and intercept assumed to evolve independently

    # Kalman filter setup
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=[0, 0],  # Start with a hedge ratio of 0 and no intercept
        initial_state_covariance=np.ones((2, 2)),
        observation_covariance=1.0,
        transition_covariance=np.ones((2, 2)) * 0.01  # Small noise in hedge ratio evolution
    )

    # Apply the Kalman filter to the log prices
    state_means, state_covariances = kf.filter(asset1_prices.values)

    # Extract the hedge ratios (state_means[:, 0] contains the dynamic hedge ratio)
    hedge_ratios = state_means[:, 0]
    intercepts = state_means[:, 1]
    
    return np.mean(hedge_ratios), hedge_ratios

def construct_spread(asset1_prices, asset2_prices, hedge_ratio):
    spread = asset1_prices - hedge_ratio * asset2_prices
    return spread

def calculate_half_life(spread):
    # Remove any NaN values from the spread
    spread = spread.dropna()

    # Lag the spread (Spread_t-1)
    lagged_spread = spread.shift(1).dropna()
    
    # Align spread and lagged_spread by dropping the first observation in spread
    spread_aligned = spread.iloc[1:]  # Drop the first row from the original spread to align with lagged_spread

    # Reshape for sklearn linear regression
    lagged_spread_reshaped = lagged_spread.values.reshape(-1, 1)  # Independent variable (X)
    spread_aligned_reshaped = spread_aligned.values  # Dependent variable (y)

    # Perform linear regression: Spread_t = beta * Spread_(t-1) + intercept + epsilon
    model = LinearRegression()
    model.fit(lagged_spread_reshaped, spread_aligned_reshaped)

    # Extract the beta coefficient
    beta = model.coef_[0]

    # Ensure valid beta
    if beta >= 1 or beta <= 0:
        return None

    # Calculate the half-life of mean reversion
    half_life = -np.log(2) / np.log(beta)
    return half_life

def traditional_metrics(i, start_date, end_date, index):
    sym_1 = i[0]
    sym_2 = i[1]

    df_1 = dm.get_sym_dates_index(index, sym_1, start_date, end_date)
    df_1['ret'] = df_1.adj_close.pct_change()
    df_1['ret'] = np.log(df_1.ret+1)
    
    df_2 = dm.get_sym_dates_index(index, sym_2, start_date, end_date)
    df_2['ret'] = df_2.adj_close.pct_change()
    df_2['ret'] = np.log(df_2.ret+1)

    df = pd.merge(df_1, df_2, on='date', suffixes=[f"_{sym_1}", f"_{sym_2}"])
    df.dropna(inplace=True)

    ret_cols = [i for i in df.columns if 'ret' in i] 
    close_cols = [i for i in df.columns if 'adj_close'in i ]

    # Pearson
    pearson = df[ret_cols].corr()[ret_cols[0]][ret_cols[1]]

    #spearman
    spearman = df[ret_cols].corr(method='spearman')[ret_cols[0]][ret_cols[1]]

    # mean reversion speed
    hedge_ratio_mean, hedge_ratios = estimate_hedge_ratio(df[close_cols[0]], df[close_cols[1]])

    spread = construct_spread(df[close_cols[0]], df[close_cols[1]], hedge_ratios)
    half_life = calculate_half_life(spread)
    
    # Cointegration  
    _, p_value, _ = coint(df[close_cols[0]], df[close_cols[1]])

    # Spread stationarity (ADF)
    adf_stats = adfuller(spread)
    t_val = adf_stats[0]
    t_stat_1_pct = adf_stats[4]['1%']
    if t_val < t_stat_1_pct:
        is_stationary = 1
    else:
        is_stationary = 0

    # Construct metric df
    metrics = {'index': index, 
               'sym_1':  sym_1, 
               'sym_2': sym_2,
               'pearson': pearson, 
               'spearman': spearman, 
               'hedge_ratio': hedge_ratio_mean, 
               'half_life': half_life, 
               'coint_pval': p_value, 
               'spread_stationary': is_stationary}
    
    metrics = pd.DataFrame(metrics, index=[0])

    return metrics

def main(index, start_date, end_date):
    symbols = dm.get_symbols(index)
    pairs = [i for i in itertools.combinations(symbols, 2)]
    # Selecting random 200 pairs - for speed
    pairs = random.sample(pairs, 200)
    metrics_df = []
    for i in tqdm(pairs):
        tmp = traditional_metrics(i, start_date, end_date, index)
        if type(tmp) == pd.DataFrame:
            metrics_df.append(tmp)
    metrics_df = pd.concat(metrics_df)
    metrics_df.dropna(inplace=True)
    metrics_df.to_csv("traditional_metrics.csv", index=False)

if __name__ == '__main__':
    print(os.getcwd())
    parser = argparse.ArgumentParser(description="Get pairs for selected index and date range")
    parser.add_argument('index', type=str, help='The index to use (Nasdaq 100 or S&P 500)')
    parser.add_argument('start_date', type=str, help='The start date (YYYY-MM-DD)')
    parser.add_argument('end_date', type=str, help='The end date (YYYY-MM-DD)')

    # Parse arguments
    args = parser.parse_args()

    index = args.index
    start_date = args.start_date
    end_date = args.end_date
    # Call the main function with parsed arguments
    main(index, start_date, end_date)
    
    """
    index = 'Nasdaq 100'
    start_date = '2020-09-21'
    end_date = '2024-09-21'
    main(index, start_date, end_date)
    """


