import pandas as pd
import numpy as np
import sqlite3
import data_manager as dm
from get_pairs_traditional import construct_spread, estimate_hedge_ratio, calculate_half_life
import argparse

def main(index, sym_1, sym_2, start_date, end_date, window):
    # adjust start date by window
    start_date_org = start_date
    start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=window*5)).strftime('%Y-%m-%d')
    print(start_date, start_date_org, end_date)
    df_1 = dm.get_sym_dates_index(index, sym_1, start_date, end_date)
    df_2 = dm.get_sym_dates_index(index, sym_2, start_date, end_date)
    df = pd.merge(df_1, df_2, on=['date'], suffixes=('_sym_1','_sym_2'))
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    # Get hedge ratio
    hedge_ratio_mean, hedge_ratios = estimate_hedge_ratio(df.adj_close_sym_1, df.adj_close_sym_2)
    df['hedge_ratio'] = hedge_ratios

    df.dropna(inplace=True)

    # Construct spread
    df['spread'] = construct_spread(df.adj_close_sym_1, df.adj_close_sym_2, df.hedge_ratio)
    df['spread_zscore'] = (df.spread - df.spread.rolling(window=window, min_periods=window).mean())/df.spread.rolling(window=window, min_periods=window).std()
    df = df[df.date >= pd.to_datetime(start_date_org)]
    df.dropna(inplace=True)
    df = df[['date','adj_close_sym_1', 'adj_close_sym_2', 'spread_zscore', 'hedge_ratio']]
    df.to_csv('traditional_strategy.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get pairs for selected index and date range")
    parser.add_argument('index', type=str, help='The index to use (Nasdaq 100 or S&P 500)')
    parser.add_argument('start_date', type=str, help='The start date (YYYY-MM-DD)')
    parser.add_argument('end_date', type=str, help='The end date (YYYY-MM-DD)')
    parser.add_argument('sym_1', type=str, help='Symbol 1')
    parser.add_argument('sym_2', type=str, help='sym_2')
    parser.add_argument('window', type=int, help='window length')

    # Parse arguments
    args = parser.parse_args()

    index = args.index
    start_date = args.start_date
    end_date = args.end_date
    sym_1 = args.sym_1
    sym_2 = args.sym_2
    window = args.window

    # Call the main function with parsed arguments
    main(index, sym_1, sym_2, start_date, end_date, window)

    """
    index = 'Nasdaq 100'
    start_date = '2020-09-21'
    end_date = '2024-09-21'
    sym_1 = 'MSFT'
    sym_2 = 'AMZN'
    window = 90
    main(index, sym_1, sym_2, start_date, end_date, window)
    """