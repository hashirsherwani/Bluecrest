import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import sqlite3
import itertools
import networkx.algorithms.community as nx_comm
from tqdm import tqdm
from get_pairs_traditional import traditional_metrics
import data_manager as dm
import argparse
import random
random.seed(1)
np.random.seed(1)
import os

cwd = os.getcwd()
if 'project' in cwd:
    conn = sqlite3.connect('..\\equities_data.db')
else:    
    # Connect to the SQLite database
    conn = sqlite3.connect('equities_data.db')

#conn = sqlite3.connect('equities_data.db')

def calculate_euclidean_distance_matrix(df):
    # Compute the Euclidean distance between the rows (stocks) in the DataFrame
    distance_matrix = pdist(df.T, metric='euclidean')  # .T transposes the DataFrame to compare columns (stocks)
    
    # Convert the condensed distance matrix into a square form
    distance_matrix = squareform(distance_matrix)
    
    return pd.DataFrame(distance_matrix, index=df.columns, columns=df.columns)

def create_graph_from_distance(distance_matrix, threshold=0.5):
    G = nx.Graph()

    # Add nodes (symbols)
    symbols = distance_matrix.columns
    G.add_nodes_from(symbols)

    # Add edges (pairs) based on distances below the threshold (closer stocks)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            if distance_matrix.iloc[i, j] < threshold:  # Lower distance means higher similarity
                G.add_edge(symbols[i], symbols[j], weight=distance_matrix.iloc[i, j])

    return G
    
def main(index, start_date, end_date):
    df = dm.get_all_dates_index(index, start_date, end_date)

    rets = df.groupby('symbol').apply(lambda x: x.sort_values('date').adj_close.pct_change()).reset_index()
    rets = rets.set_index('level_1')
    df = pd.merge(df, rets, left_index=True, right_index=True)
    df = df[['date','symbol_y','adj_close_y']]
    df.columns = ['date','symbol','ret']
    df['ret'] = np.log(df.ret + 1)
   
    df.dropna(inplace=True)
    df = df.pivot(index='date', columns='symbol', values='ret')

    distance_matrix = calculate_euclidean_distance_matrix(df)

    # Use bottom quintile as threshold
    threshold = distance_matrix.mean(axis=1).quantile(0.5)
    print(threshold)
    G = create_graph_from_distance(distance_matrix, threshold=threshold)

    communities = nx_comm.louvain_communities(G, weight='weight')

    max_comm_size = 0
    max_comm = []
    for i in communities:
        if len(i) > max_comm_size:
            max_comm_size = len(i)
            max_comm = i

    print(f"Largest community size: {len(max_comm)}")

    # Save full community
    with open('full_community.txt', mode='w') as f:
        for line in max_comm:
            f.write(f"{line}\n")

    # Find all pair combinations for the largest community.
    pairs = [i for i in itertools.combinations(max_comm, 2)]
    final = []
    for i in tqdm(pairs):
        final.append(traditional_metrics(i, start_date, end_date, index))
    final = pd.concat(final)

    final = final[(final.coint_pval < 0.05) & (final.spread_stationary == 1)]

    final.dropna(inplace=True)
    # Save graph extracted pairs.
    final.to_csv('pairs_graph_extracted.csv', index=False)

if __name__ == '__main__':
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
    start_date = '2007-01-01'
    end_date = '2010-01-01'
    main(index, start_date, end_date)
    """