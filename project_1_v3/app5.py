import streamlit as st
import pandas as pd
import os
import subprocess  # Use subprocess to run external scripts
from sqlalchemy import create_engine
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import matplotlib.pyplot as plt

# Create the SQLAlchemy engine for SQLite
engine = create_engine('sqlite:///equities_data.db')

# Function to get unique symbols from a specific table
def get_symbols(index):
    query = ""
    if index == 'Nasdaq 100':
        query = "SELECT DISTINCT symbol FROM nasdaq100"
    elif index == 'S&P 500':
        query = "SELECT DISTINCT symbol FROM sp500"
    symbols = pd.read_sql(query, engine)
    return symbols['symbol'].tolist()

# Function to fetch data from the SQLite database
def fetch_data(symbol, index, start_date, end_date, plot_field):
    query = ""
    if index == 'Nasdaq 100':
        query = f"""
            SELECT date, {plot_field} FROM nasdaq100
            WHERE symbol = '{symbol}' AND date BETWEEN '{start_date}' AND '{end_date}'
        """
    elif index == 'S&P 500':
        query = f"""
            SELECT date, {plot_field} FROM sp500
            WHERE symbol = '{symbol}' AND date BETWEEN '{start_date}' AND '{end_date}'
        """
    data = pd.read_sql(query, engine)
    return data

# App Title
st.title("Stock Data Visualizer")

# Dropdown for selecting index
index_options = ['Nasdaq 100', 'S&P 500']
selected_index = st.selectbox('Select Index', index_options)

# Based on index selection, populate symbols dropdown
if selected_index:
    symbols = get_symbols(selected_index)
    selected_symbol = st.selectbox('Select Symbol', symbols)

# Date selection
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

# Dropdown to select the field to plot
plot_field = st.selectbox('Select Field to Plot', ['open', 'high', 'low', 'close', 'volume'])

# Fetch data and plot the graph if all inputs are selected
if selected_symbol and start_date and end_date:
    # Fetch the data
    data = fetch_data(selected_symbol, selected_index, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), plot_field)
    
    # Plotting
    if not data.empty:
        st.line_chart(data.set_index('date')[plot_field])
    else:
        st.write("No data found for the selected date range.")

# --- New Section: Pair Selection ---

st.header("Pair Selection")
st.subheader("Traditional Approach")

# Pair Selection Inputs (Start Date, End Date, Index)
pair_index = st.selectbox('Select Index for Pair Selection', index_options, key='pair_selection')
pair_start_date = st.date_input('Start Date for Pair Selection', key='pair_start_date')
pair_end_date = st.date_input('End Date for Pair Selection', key='pair_end_date')

# Execute button to run the main function from get_pairs_traditional.py
if st.button('Get top pairs', key='pair_button'):
    # Show a spinner while the script is running
    with st.spinner('Running script...'):
        try:
            # Format dates as "YYYY-MM-DD"
            formatted_pair_start_date = pair_start_date.strftime("%Y-%m-%d")
            formatted_pair_end_date = pair_end_date.strftime("%Y-%m-%d")

            # Run the external Python script using subprocess
            subprocess.run(
                ["python", "project_1_v3\\get_pairs_traditional.py", pair_index, formatted_pair_start_date, formatted_pair_end_date],
                check=True
            )
            # Check if the CSV file exists and display it
            csv_file = 'traditional_metrics.csv'
            if os.path.exists(csv_file):
                # Load the full CSV into a DataFrame and store in session state
                df = pd.read_csv(csv_file)
                st.session_state['df_traditional'] = df  # Save the DataFrame in session state

                st.write("Top 10 Rows of Sorted Pair Metrics:")
            else:
                st.write("The CSV file has not been generated yet.")

            st.success('Script ran successfully!')
        
        except subprocess.CalledProcessError as e:
            st.error(f"Error occurred while running the script: {e}")

# Display the table if it exists in session state
if 'df_traditional' in st.session_state:
    df = st.session_state['df_traditional']

    # Configure AgGrid options to allow sorting, filtering, and row selection
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationPageSize=10)  # Display only 10 rows at a time
    gb.configure_default_column(sortable=True, filterable=True)
    gb.configure_selection('multiple', use_checkbox=True)  # Enable row selection with checkboxes

    grid_options = gb.build()

    # Display the grid with selection enabled
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        height=300,
        width='100%',
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        theme='streamlit',
        key='grid_traditional',
    )

    # Extract selected rows
    selected_rows = pd.DataFrame(grid_response['selected_rows'])

    # Display button to download the selected rows as CSV
    if not selected_rows.empty:
        st.write(f"{len(selected_rows)} rows selected.")
        st.download_button(
            label="Download Selected Rows as CSV",
            data=selected_rows.to_csv(index=False),
            file_name='user_selected_traditional.csv',
            mime='text/csv'
        )

# --- New Section: Graph Pair Selection ---

st.header("Pair Selection - Graph/Network")
st.subheader("Alternative Approach")
st.text("Extracted pairs are filtered on coint p_value < 0.05 and half life of <= 30 days")

pair_index_alt = st.selectbox('Select Index for Pair Selection', index_options, key='pair_selection_alt')
pair_start_date_alt = st.date_input('Start Date for Pair Selection', key='pair_start_date_alt')
pair_end_date_alt = st.date_input('End Date for Pair Selection', key='pair_end_date_alt')

if st.button('Extract Pairs', key='alt_approach'):
    # Show a spinner while the script is running
    with st.spinner('Running script...'):
        try:
            # Format dates as "YYYY-MM-DD"
            formatted_pair_start_date_alt = pair_start_date_alt.strftime("%Y-%m-%d")
            formatted_pair_end_date_alt = pair_end_date_alt.strftime("%Y-%m-%d")

            # Run the external Python script using subprocess
            subprocess.run(
                ["python", "project_1_v3\\get_graph_pairs.py", pair_index_alt, formatted_pair_start_date_alt, formatted_pair_end_date_alt], 
                check=True
            )
            # Check if the CSV file exists and display it
            csv_file_alt = 'pairs_graph_extracted.csv'
            if os.path.exists(csv_file_alt):
                # Load the full CSV into a DataFrame and store in session state
                df = pd.read_csv(csv_file_alt)
                st.session_state['df_alternative'] = df  # Save the DataFrame in session state

                st.write("Top 10 Rows of Sorted Pair Metrics:")
            else:
                st.write("The CSV file has not been generated yet.")

            st.success('Script ran successfully!')
        
        except subprocess.CalledProcessError as e:
            st.error(f"Error occurred while running the script: {e}")

# Display the table if it exists in session state
if 'df_alternative' in st.session_state:
    df = st.session_state['df_alternative']

    # Configure AgGrid options to allow sorting, filtering, and row selection
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationPageSize=10)  # Display only 10 rows at a time
    gb.configure_default_column(sortable=True, filterable=True)
    gb.configure_selection('multiple', use_checkbox=True)  # Enable row selection with checkboxes

    grid_options = gb.build()

    # Display the grid with selection enabled
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        height=300,
        width='100%',
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        theme='streamlit',
        key='grid_alternative',
    )

    # Extract selected rows
    selected_rows = pd.DataFrame(grid_response['selected_rows'])

    # Display button to download the selected rows as CSV
    if not selected_rows.empty:
        st.write(f"{len(selected_rows)} rows selected.")
        st.download_button(
            label="Download Selected Rows as CSV",
            data=selected_rows.to_csv(index=False),
            file_name='user_selected_alternative.csv',
            mime='text/csv'
        )

# --- New Section: Traditional Strategy ---

st.header("Traditional Strategy")

# User input for index, start date, and end date
strategy_index = st.selectbox('Select Index for Traditional Strategy', index_options, key='traditional_strategy_index')
strategy_start_date = st.date_input('Start Date (In-Sample)', key='traditional_strategy_start')
strategy_end_date = st.date_input('End Date for (In-Sample)', key='traditional_strategy_end')
strategy_start_date_os = st.date_input('Start Date (Out-Sample)', key='traditional_strategy_start_os')
strategy_end_date_os = st.date_input('End Date for (Out-Sample)', key='traditional_strategy_end_os')

# Text inputs for symbols and window
sym_1 = st.text_input('Enter Symbol 1 (sym_1)', key='traditional_strategy_sym_1')
sym_2 = st.text_input('Enter Symbol 2 (sym_2)', key='traditional_strategy_sym_2')
window = st.number_input('Enter Window (int)', min_value=1, step=1, key='traditional_strategy_window')

col1, col2 = st.columns([1, 1])

# Button to execute traditional_strategy.py
with col1:
    if st.button('Form Strategy: In-Sample', key='traditional_strategy'):
        # Show a spinner while the script is running
        with st.spinner('Running traditional strategy...'):
            try:
                # Format dates as "YYYY-MM-DD"
                formatted_strategy_start_date = strategy_start_date.strftime("%Y-%m-%d")
                formatted_strategy_end_date = strategy_end_date.strftime("%Y-%m-%d")

                # Run the external Python script using subprocess
                subprocess.run(
                    ["python", "project_1_v3\\traditional_strategy.py", strategy_index, formatted_strategy_start_date, formatted_strategy_end_date, sym_1, sym_2, str(window)], 
                    check=True
                )
                st.success('Traditional Strategy ran successfully!')
            
                # Load the CSV file generated by the script
                csv_file = 'traditional_strategy.csv'
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)

                    # Persist the user-selected columns in session state
                    columns = [col for col in df.columns if col != 'date']
                    if 'selected_columns' not in st.session_state:
                        st.session_state.selected_columns = []

                    # Allow the user to select columns for plotting (excluding the date column)
                    selected_columns = st.multiselect('Select columns to plot', columns, default=st.session_state.selected_columns, key='plot_columns')
                    st.session_state.selected_columns = selected_columns

                    if selected_columns:
                        # Plot with dual y-axis if more than one column is selected
                        fig, ax1 = plt.subplots()

                        # Always use the date column as the x-axis
                        df['date'] = pd.to_datetime(df['date'])
                        ax1.set_xlabel('Date')

                        # Plot the first column on the left y-axis
                        ax1.plot(df['date'], df[selected_columns[0]], 'b-', label=selected_columns[0])
                        ax1.set_ylabel(selected_columns[0], color='b')
                        ax1.tick_params('y', colors='b')

                        # Plot the second column (if any) on the right y-axis
                        if len(selected_columns) > 1:
                            ax2 = ax1.twinx()
                            ax2.plot(df['date'], df[selected_columns[1]], 'r-', label=selected_columns[1])
                            ax2.set_ylabel(selected_columns[1], color='r')
                            ax2.tick_params('y', colors='r')

                        # Add legends
                        ax1.legend(loc='upper left')
                        if len(selected_columns) > 1:
                            ax2.legend(loc='upper right')

                        # Improve layout
                        fig.tight_layout()

                        # Display the plot
                        st.pyplot(fig)

                    else:
                        st.write("Please select at least one column to plot.")

                else:
                    st.write("The traditional_strategy.csv file was not found.")
            
            except subprocess.CalledProcessError as e:
                st.error(f"Error occurred while running the traditional strategy script: {e}")

    # If the CSV file exists (even after initial run), allow plotting
    elif os.path.exists('traditional_strategy.csv'):
        df = pd.read_csv('traditional_strategy.csv')

        # Persist the user-selected columns in session state
        columns = [col for col in df.columns if col != 'date']
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = []

        # Allow the user to select columns for plotting (excluding the date column)
        selected_columns = st.multiselect('Select columns to plot', columns, default=st.session_state.selected_columns, key='plot_columns')
        st.session_state.selected_columns = selected_columns

        if selected_columns:
            # Plot with dual y-axis if more than one column is selected
            fig, ax1 = plt.subplots()

            # Always use the date column as the x-axis
            df['date'] = pd.to_datetime(df['date'])
            ax1.set_xlabel('Date')

            # Plot the first column on the left y-axis
            ax1.plot(df['date'], df[selected_columns[0]], 'b-', label=selected_columns[0])
            ax1.set_ylabel(selected_columns[0], color='b')
            ax1.tick_params('y', colors='b')

            # Plot the second column (if any) on the right y-axis
            if len(selected_columns) > 1:
                ax2 = ax1.twinx()
                ax2.plot(df['date'], df[selected_columns[1]], 'r-', label=selected_columns[1])
                ax2.set_ylabel(selected_columns[1], color='r')
                ax2.tick_params('y', colors='r')

            # Add legends
            ax1.legend(loc='upper left')
            if len(selected_columns) > 1:
                ax2.legend(loc='upper right')

            # Improve layout
            fig.tight_layout()

            # Display the plot
            st.pyplot(fig)

        else:
            st.write("Please select at least one column to plot.")

with col2:
    if st.button('Form Strategy: Out-Sample', key='traditional_strategy_os'):
        # Show a spinner while the script is running
        with st.spinner('Running traditional strategy...'):
            try:
                # Format dates as "YYYY-MM-DD"
                formatted_strategy_start_date_os = strategy_start_date_os.strftime("%Y-%m-%d")
                formatted_strategy_end_date_os = strategy_end_date_os.strftime("%Y-%m-%d")

                # Run the external Python script using subprocess
                subprocess.run(
                    ["python", "project_1_v3\\traditional_strategy.py", strategy_index, formatted_strategy_start_date_os, formatted_strategy_end_date_os, sym_1, sym_2, str(window)], 
                    check=True
                )
                st.success('Traditional Strategy (out-sample) ran successfully!')
            except subprocess.CalledProcessError as e:
                st.error(f"Error occurred while running the traditional strategy script: {e}")

# --- New Section: Traditional Backtest ---

st.subheader("Traditional Backtest")

# User inputs for Entry and Exit thresholds
entry_threshold = st.number_input('Entry Threshold (zscore)', value=1.0, step=0.1, key='entry_threshold')
exit_threshold = st.number_input('Exit Threshold (zscore)', value=0.0, step=0.1, key='exit_threshold')

col1, col2 = st.columns([1, 1])  # Create two columns for the buttons

with col1:
    # Run Backtest Button
    if st.button('Run Backtest', key='run_backtest'):
        with st.spinner('Running backtest...'):
            try:
                # Ensure the traditional_strategy.csv file exists
                if os.path.exists('traditional_strategy.csv'):
                    # Run the backtester.py script with the entry and exit thresholds
                    subprocess.run(
                        ["python", "project_1_v3\\backtester.py", 'traditional_strategy.csv', 'False', str(entry_threshold), str(exit_threshold)], 
                        check=True
                    )

                    st.success('Backtest ran successfully!')

                    # Check if the 'traditional_strategy_perf.csv' file was generated
                    perf_csv_file = 'traditional_strategy_perf.csv'
                    if os.path.exists(perf_csv_file):
                        # Load the performance CSV into a DataFrame
                        perf_df = pd.read_csv(perf_csv_file)

                        # Display the performance data using AgGrid
                        st.write("Performance Data from Backtest:")
                        gb = GridOptionsBuilder.from_dataframe(perf_df)
                        gb.configure_pagination(paginationPageSize=10)
                        gb.configure_default_column(sortable=True, filterable=True)
                        grid_options = gb.build()

                        AgGrid(
                            perf_df,
                            gridOptions=grid_options,
                            height=300,
                            width='100%',
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            theme='streamlit',
                            key='grid_perf',
                        )
                    else:
                        st.error("The traditional_strategy_perf.csv file was not found.")

                else:
                    st.error('The traditional_strategy.csv file was not found.')

            except subprocess.CalledProcessError as e:
                st.error(f"Error occurred while running the backtest: {e}")

with col2:    
    if st.button('Run Optimizer', key='run_optimizer'):
        with st.spinner('Running optimizer...'):
            try:
                # Run the optimizer.py script and capture the output
                result = subprocess.run(
                    ["python", "project_1_v3\\backtester.py", 'traditional_strategy.csv', 'True', str(entry_threshold), str(exit_threshold)],
                    capture_output=True, text=True, check=True
                )
                
                # Extract the stdout and find the relevant lines (best_thresholds and best_sharpe)
                output = result.stdout
                
                for line in output.splitlines():
                    if "best_thresholds" in line:
                        best_thresholds = line
                    if "best_sharpe" in line:
                        best_sharpe = line

                # Display the results in st.success
                if best_thresholds and best_sharpe:
                    st.success(f"Optimizer ran successfully!\n{best_thresholds}\n{best_sharpe}")
                    #st.success(f"Optimizer ran successfully!\n{output}\n")
                else:
                    st.warning("Optimizer ran, but no best_thresholds or best_sharpe found in the output.")

            except subprocess.CalledProcessError as e:
                st.error(f"Error occurred while running the optimizer: {e}")