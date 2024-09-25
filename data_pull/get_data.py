import yfinance as yf
import pandas as pd
import sqlite3
from tqdm import tqdm

# Function to get S&P 500 components from Wikipedia
def get_sp500_components():
    url = f'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tickers = pd.read_html(url)[0]
    tickers = tickers.Symbol.to_list()
    return tickers

# Function to get Nasdaq 100 components from Wikipedia
def get_nasdaq100_components():
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    tickers = pd.read_html(url)[4]
    tickers = tickers.Ticker.to_list()
    return tickers

# Function to download data for a given list of tickers (Open, High, Low, Close, Volume, Adj Close)
def download_data(tickers, start_date="2007-01-01"):
    if not tickers:
        print("No tickers to download.")
        return pd.DataFrame()
    
    data = yf.download(tickers, start=start_date, progress=False)
    return data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# Function to reshape data from wide format to long format (handle MultiIndex or single-index)
def reshape_data(tickers, data):
    # Check if data is MultiIndex (multiple tickers) or single-index (one ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # Create an empty list to hold each ticker's data
        frames = []
        
        # Iterate through each ticker and its corresponding data
        for ticker in tickers:
            # Extract the data for the current ticker
            ticker_data = data.xs(ticker, level=1, axis=1).copy()
            
            # Add a 'symbol' column with the ticker symbol
            ticker_data['symbol'] = ticker
            
            # Append to the list of frames
            frames.append(ticker_data)
        
        # Concatenate all the frames into a single DataFrame
        combined_data = pd.concat(frames)
    else:
        # If data is single-index (only one ticker), add a 'symbol' column
        combined_data = data.copy()
        combined_data['symbol'] = tickers[0]  # Assume there's only one ticker

    # Reset the index to turn 'Date' into a column
    combined_data.reset_index(inplace=True)
    
    # Rename columns to lowercase and remove spaces
    combined_data.columns = [col.lower().replace(' ', '_') for col in combined_data.columns]
    
    return combined_data

# Function to store data into SQLite database
def store_data_in_sqlite(table_name, data, conn):
    if data.empty:
        print(f"No data available for {table_name}, skipping storage.")
        return
    
    data['date'] = data.date.apply(lambda x: x.date())
    # Insert data into SQL with lowercase and no space in column names
    data.to_sql(table_name, conn, if_exists='replace', index=False)

# Main function to download data and store into SQLite with progress tracking
def download_and_store_data(db_path='equities_data.db', start_date='2007-01-01'):
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)

    # Get tickers for S&P 500 and Nasdaq 100
    print("Fetching tickers for S&P 500 and Nasdaq 100...")
    sp500_tickers = get_sp500_components()
    nasdaq100_tickers = get_nasdaq100_components()

    if not sp500_tickers:
        print("Failed to fetch S&P 500 tickers.")
    if not nasdaq100_tickers:
        print("Failed to fetch Nasdaq 100 tickers.")

    # Indices tickers
    indices = {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX"
    }

    # Download data with progress tracking for each group
    print("Downloading S&P 500 components data...")
    sp500_raw_data = download_data(sp500_tickers)
    sp500_data = reshape_data(sp500_tickers, sp500_raw_data)
    
    print("Downloading Nasdaq 100 components data...")
    nasdaq100_raw_data = download_data(nasdaq100_tickers)
    nasdaq100_data = reshape_data(nasdaq100_tickers, nasdaq100_raw_data)

    # Download indices performance
    index_data = {}
    print("Downloading index performance data...")
    for index_name, ticker in tqdm(indices.items(), desc="Downloading indices"):
        index_data[index_name] = download_data([ticker])

    # Store data in SQLite with progress tracking
    print("Storing S&P 500 data into SQLite database...")
    store_data_in_sqlite('sp500', sp500_data, conn)

    print("Storing Nasdaq 100 data into SQLite database...")
    store_data_in_sqlite('nasdaq100', nasdaq100_data, conn)

    print("Storing index performance data into SQLite database...")
    for index_name, data in tqdm(index_data.items(), desc="Storing indices"):
        data = reshape_data([ticker], data)
        if 'Nasdaq' in index_name:
            store_data_in_sqlite('nasdaq100_index', data, conn)
        elif 'S&P' in index_name:
            store_data_in_sqlite('sp500_index', data, conn)

    # Commit and close connection
    conn.commit()
    conn.close()

    print(f"Data successfully downloaded and stored in {db_path}")

# Example usage
download_and_store_data()
