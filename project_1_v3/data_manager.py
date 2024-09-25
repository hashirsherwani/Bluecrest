import sqlite3
import pandas as pd
import os

cwd = os.getcwd()

if 'project' in cwd:
    conn = sqlite3.connect('..\\equities_data.db')
else:    
    # Connect to the SQLite database
    conn = sqlite3.connect('equities_data.db')

# Get unique symbols for a given index
def get_symbols(index):
    if index == 'Nasdaq 100':
        query = "SELECT DISTINCT symbol FROM nasdaq100"
    elif index == 'S&P 500':
        query = "SELECT DISTINCT symbol FROM sp500"
    symbols = pd.read_sql(query, conn)
    return symbols['symbol'].tolist()

def get_sym_dates_index(index, symbol, start_date, end_date):
    if index == 'Nasdaq 100':
        query = f"SELECT * FROM nasdaq100 where symbol = '{symbol}' AND date BETWEEN '{start_date}' AND '{end_date}'"
    elif index == 'S&P 500':
        query = f"SELECT * FROM sp500 where symbol = '{symbol}' AND date BETWEEN '{start_date}' AND '{end_date}'"
    data = pd.read_sql(query, conn)
    data['date'] = pd.to_datetime(data.date)
    return data

def get_all_dates_index(index, start_date, end_date):
    if index == 'Nasdaq 100':
        query = f"SELECT date,symbol,adj_close FROM nasdaq100 where date BETWEEN '{start_date}' AND '{end_date}'"
    elif index == 'S&P 500':
        query = f"SELECT date,symbol,adj_close FROM sp500 where date BETWEEN '{start_date}' AND '{end_date}'"
    data = pd.read_sql(query, conn)
    data['date'] = pd.to_datetime(data.date)
    data.dropna(inplace=True)
    return data


if __name__ == '__main__':
    pass