import pandas as pd
import re
import glob
import os
import json

def clean_datetime_string(date_time_string):
    """
    Remove the decimal part from the seconds in the datetime string 
    to ensure proper formatting for conversion to datetime.

    Parameters:
    date_time_string (str): The datetime string to clean.

    Returns:
    str: The cleaned datetime string.

    Raises:
    ValueError: If the input is not a string.
    """
    if not isinstance(date_time_string, str):
        raise ValueError("Input must be a string")
    cleaned_string = re.sub(r'\.\d+', '', date_time_string)
    return cleaned_string

def update_dataframe(main_df, new_df):
    """
    Update the main dataframe with new data from another dataframe.
    Ensures that datetime columns are properly formatted and merged.

    Parameters:
    main_df (pd.DataFrame): The main dataframe to be updated.
    new_df (pd.DataFrame): The new dataframe with additional data.

    Returns:
    pd.DataFrame: The updated dataframe.

    Raises:
    ValueError: If the 'Time' column is not present in either dataframe.
    """
    # Check if 'Time' column is present in both dataframes
    if 'Time' not in main_df.columns or 'Time' not in new_df.columns:
        raise ValueError("Both dataframes must contain a 'Time' column.")

    # Copy dataframes to avoid modifying the original ones
    df = main_df.copy()
    dff = new_df.copy()
    try:
        # Preparation of datetime strings to be convertible
        df['Time'] = df['Time'].apply(clean_datetime_string)
        dff['Time'] = dff['Time'].apply(clean_datetime_string)
        # Conversion of time column into datetime
        df['Time'] = pd.to_datetime(df['Time'])
        dff['Time'] = pd.to_datetime(dff['Time'])
    except Exception as e:
        raise ValueError(f"Error in datetime conversion: {e}")

    # Estimate min and max datetimes for function logic
    df_time_last = df['Time'].max()
    dff_time_first = dff['Time'].min()
    dff_time_last = dff['Time'].max()
    # Logic for concat operation
    if df_time_last < dff_time_first:
        # Merge whole dataframes
        df = pd.concat([df, dff], axis=0)
    elif df_time_last < dff_time_last:
        # Filter dates in new dataframe before merge
        dff_filtered = dff[dff['Time'] > df_time_last]
        df = pd.concat([df, dff_filtered], axis=0)
    else:
        print("First dataframe is up to date")

    # Convert 'Time' column back to string format
    df['Time'] = df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df

def update_main_csv(origin_csv_file, path_to_csv_files):
    """
    Update the main CSV file using other CSV files from 
    a specified directory. Ensures that all datetime columns 
    are properly formatted and merged.

    Parameters:
    origin_csv_file (str): Path to the main CSV file.
    path_to_csv_files (str): Path to the directory containing
    the CSV files to merge.

    Returns:
    None

    Raises:
    FileNotFoundError: If the origin CSV file or directory of 
    CSV files does not exist.ValueError: If there is an error in 
    reading or processing the CSV files.
    """
    try:
        # Check if the main CSV file exists
        if not os.path.isfile(origin_csv_file):
            raise FileNotFoundError(f"Origin CSV file " +
                                    "'{origin_csv_file}'" +
                                    "does not exist.")
        # Check if the directory containing the CSV files exists
        if not os.path.isdir(path_to_csv_files):
            raise FileNotFoundError(f"Directory " +
                                    "'{path_to_csv_files}'" +
                                    "does not exist.")
        # Get list of all CSV files in the specified directory
        path = os.path.join(path_to_csv_files, '*.csv')
        csv_files = glob.glob(path)
        # Read the main CSV file
        main = pd.read_csv(origin_csv_file)
        # Update the main dataframe with each new CSV file
        for file in csv_files:
            new = pd.read_csv(file)
            main = update_dataframe(main, new)
        # Save the updated main dataframe back to the CSV file
        main.to_csv(origin_csv_file, index=False)
        print(f"Main CSV file '{origin_csv_file}' "+
              "has been updated.")
    except Exception as e:
        raise ValueError(f"An error occurred while" +
                         "updating the main CSV file: {e}")

def total_deposit(df):
    """
    Calculate the total deposit amount from a Trading212 app 
    dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame containing trading data with 
    at least 'Action' and 'Total' columns.

    Returns:
    float: The total deposit amount.

    Raises:
    ValueError: If the required columns are not present in
     the DataFrame.
    """
    # Check if the required columns are present in the DataFrame
    if 'Action' not in df.columns or 'Total' not in df.columns:
        raise ValueError("DataFrame must contain " +
                         "'Action' and 'Total' columns.")
    # Filter the DataFrame to include only rows where the
    # 'Action' is 'Deposit'
    deposit_df = df[df['Action'] == 'Deposit']['Total']
    # Calculate the total deposit amount
    total_deposit = deposit_df.sum()
    return total_deposit

def total_withdrawal(df):
    """
    Calculate the total deposit amount from a Trading212 
    app dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame containing trading data 
    with at least 'Action' and 'Total' columns.

    Returns:
    float: The total deposit amount.

    Raises:
    ValueError: If the required columns are not present 
    in the DataFrame.
    """
    # Check if the required columns are present in the DataFrame
    if 'Action' not in df.columns or 'Total' not in df.columns:
        raise ValueError("DataFrame must contain" +
                          "'Action' and 'Total' columns.")
    # Filter the DataFrame to include only rows where
    # the 'Action' is 'Deposit'
    withdrawal_df = df[df['Action'] == 'Withdrawal']['Total']
    # Calculate the total deposit amount
    total_withdrawal = withdrawal_df.sum()
    return total_withdrawal

def correct_ticker_for_yfinance(ticker):
    """
    Correct known issues with ticker symbols for use with 
    yfinance.

    Parameters:
    ticker (str): The original ticker symbol.

    Returns:
    str: The corrected ticker symbol if it exists in 
    known_issues, else the original ticker symbol.
    """
    known_issues = {
        'BRK.B': 'BRK-B',
        'BARC': 'BARC.L',
        'RHM': 'RHM.DE',
        'P911': 'P911.DE',
        'ASML': 'ASML.AS',
        'BP': 'BP.L',
        'SAP': 'SAP.DE',
        'MNG': "MNG.L",
        "POLR": "POLR.L",
        "IGG": "IGG.L",
        "CSN": "CSN.L",
        "MONY":"MONY.L",
        "LGEN":"LGEN.L",
        "SVT":"SVT.L",
        "TEP":"TEP.L",
        "RIO":"RIO.L"
    }
    # Check if the ticker needs correction
    if ticker in known_issues:
        corrected_ticker = known_issues[ticker]
    else:
        corrected_ticker = ticker
    return corrected_ticker

def check_known_stocksplit(ticker):
    """
    Retrieve known stock split event details for 
    a given ticker.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    tuple: A tuple containing the stock split date 
    (as a datetime object) and the stock split ratio.

    Raises:
    ValueError: If the ticker is not a string.
    """
    known_stock_splits = {
        'NVDA': {'date': '2024-06-10 00:00:00', 'ratio': 10}
    }
    if ticker in known_stock_splits:
        stock_split_date = pd.to_datetime(known_stock_splits[ticker]['date'])
        stock_split_ratio = known_stock_splits[ticker]['ratio']
    else:
        stock_split_date = pd.to_datetime('2000-01-01 00:00:00')
        stock_split_ratio = 1
    return stock_split_date, stock_split_ratio

def update_stock_dict(df):
    """
    Update the actual number of shares and titles
    in the account based on trading actions.

    Parameters:
    df (pd.DataFrame): DataFrame containing trading
    data with at least 'Action', 'Ticker',
    'No. of shares', and 'Time' columns.

    Returns:
    None

    Raises:
    ValueError: If the required columns are not present in the DataFrame.
    """
    required_columns = ['Action', 'Ticker', 'No. of shares', 'Time']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")

    # Ensure the 'Time' column is in datetime format
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    if df['Time'].isna().any():
        raise ValueError("Some dates in 'Time' column could not be converted to datetime.")

    Stock_Dict = {}
    tickers = df['Ticker'].unique()

    for ticker in tickers:
        # Initialize total shares
        total_shares = 0
        split_date, split_ratio = check_known_stocksplit(ticker)
        # Process each row for the current ticker
        for _, row in df[(df['Ticker'] == ticker) &
                         (df['Action'].isin(['Market buy', 'Market sell']))
                         ].iterrows():
            action = row['Action']
            no_of_shares = row['No. of shares']
            transaction_time = row['Time']
            # Adjust for stock splits for transactions after the split date
            if transaction_time <= split_date:
                no_of_shares *= split_ratio
            if action == 'Market buy':
                total_shares += no_of_shares
            elif action == 'Market sell':
                total_shares -= no_of_shares
        if total_shares > 5e-6:
            Stock_Dict[correct_ticker_for_yfinance(ticker)] = total_shares
    with open('Stock_Dict.json', 'w') as file:
        # Save dictionary into JSON
        json.dump(Stock_Dict, file, ensure_ascii=False, indent=4)

    print("Stock dictionary has been updated and saved to 'Stock_Dict.json'.")

def market_sell_total(df, start_date_time, end_date_time):
    """
    Returns the total sell volume in CZK for a specific time window.

    Parameters:
    df (pd.DataFrame): DataFrame containing trading data with 'Action', 'Time', and 'Total' columns.
    start_date_time (str or datetime): The start of the time window.
    end_date_time (str or datetime): The end of the time window.

    Returns:
    float: Total sell volume in CZK for the specified time window.

    Raises:
    ValueError: If the required columns are not present in the DataFrame.
    """
    required_columns = ['Action', 'Time', 'Total']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
    # Convert 'Time' column to datetime
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    if df['Time'].isna().any():
        raise ValueError("Some dates in 'Time' column could"
                         +" not be converted to datetime.")
    # Convert start_date_time and end_date_time to datetime
    start_date_time = pd.to_datetime(start_date_time, errors='coerce')
    end_date_time = pd.to_datetime(end_date_time, errors='coerce')
    if pd.isna(start_date_time) or pd.isna(end_date_time):
        raise ValueError("start_date_time and end_date_time must be valid datetime" +
                         "strings or datetime objects.")
    total_sells = df[(df['Action'] == 'Market sell') &
                     (df['Time'] >= start_date_time) &
                     (df['Time'] <= end_date_time)]['Total'].sum()
    return total_sells

def market_buy_total(df, start_date_time, end_date_time):
    """
    Returns the total buy volume in CZK for a specific time window.

    Parameters:
    df (pd.DataFrame): DataFrame containing trading data with 'Action', 'Time', and 'Total' columns.
    start_date_time (str or datetime): The start of the time window.
    end_date_time (str or datetime): The end of the time window.

    Returns:
    float: Total buy volume in CZK for the specified time window.

    Raises:
    ValueError: If the required columns are not present in the DataFrame.
    """
    required_columns = ['Action', 'Time', 'Total']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
    # Convert 'Time' column to datetime
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    if df['Time'].isna().any():
        raise ValueError("Some dates in 'Time' column could not be converted to datetime.")
    # Convert start_date_time and end_date_time to datetime
    start_date_time = pd.to_datetime(start_date_time, errors='coerce')
    end_date_time = pd.to_datetime(end_date_time, errors='coerce')
    if pd.isna(start_date_time) or pd.isna(end_date_time):
        raise ValueError("start_date_time and end_date_time must" +
                        "be valid datetime strings or datetime objects.")
    total_buys = df[(df['Action'] == 'Market buy') &
                    (df['Time'] >= start_date_time) &
                    (df['Time'] <= end_date_time)]['Total'].sum()
    return total_buys

def trading_result(df, start_date_time, end_date_time):
    """
    Returns the net result (gain or loss) in CZK from trading operations 
    for a specific time window.

    Parameters:
    df (pd.DataFrame): DataFrame containing trading data with 'Action', 
    'Time', 'Total', and 'Result' columns.
    start_date_time (str or datetime): The start of the time window.
    end_date_time (str or datetime): The end of the time window.

    Returns:
    float: Net result (gain or loss) in CZK for the specified time window.

    Raises:
    ValueError: If the required columns are not present in the DataFrame or
    if there is an error in calculating the trading result.
    """
    required_columns = ['Action', 'Time', 'Total', 'Result']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
    # Convert 'Time' column to datetime
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    if df['Time'].isna().any():
        raise ValueError("Some dates in 'Time' column could not be converted to datetime.")
    # Convert start_date_time and end_date_time to datetime
    start_date_time = pd.to_datetime(start_date_time, errors='coerce')
    end_date_time = pd.to_datetime(end_date_time, errors='coerce')
    if pd.isna(start_date_time) or pd.isna(end_date_time):
        raise ValueError("start_date_time and end_date_time must be valid datetime strings or datetime objects.")
    try:
        result = df[(df['Action'] == 'Market sell') &
                    (df['Time'] >= start_date_time) &
                    (df['Time'] <= end_date_time)]['Result'].sum()
    except Exception as e:
        raise ValueError(f"Error calculating trading result: {e}")

    return result

def total_dividends(df):
    """
    Calculate total dividends received in CZK.

    Parameters:
    df (pd.DataFrame): DataFrame containing trading data with at least 'Action' and 'Total' columns.

    Returns:
    float: Total dividends received in CZK.

    Raises:
    ValueError: If the required columns are not present in the DataFrame.
    """
    required_columns = ['Action', 'Total']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")

    # Calculate total dividends
    try:
        div_total = df[df['Action'].isin([
            'Dividend (Ordinary)',
            'Dividend (Dividends by us corporations)',
            'Dividend (Dividend)',
            'Dividend (Tax exempted)',
            'Dividend (Dividend manufactured payment)'
        ])]['Total'].sum()
    except Exception as e:
        raise ValueError(f"Error calculating total dividends: {e}")

    return div_total

def total_interest(df):
    """
    Calculate total money received from interest.

    Parameters:
    df (pd.DataFrame): DataFrame containing trading data with at least 'Action' and 'Total' columns.

    Returns:
    float: Total interest received.

    Raises:
    ValueError: If the required columns are not present in the DataFrame or if there is an error in calculating the total interest.
    """
    required_columns = ['Action', 'Total']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
        
    # Calculate total interest
    try:
        int_total = df[df['Action'].isin([
            'Lending interest',
            'Interest on cash'
        ])]['Total'].sum()
    except Exception as e:
        raise ValueError(f"Error calculating total interest: {e}")

    return int_total

def update_tax_dict(df):
    """
    Update Dividend tax dictionary into a JSON file.

    Parameters:
    df (pd.DataFrame): DataFrame containing trading data with at least 'Action', 'Ticker', 'Time',
                       'No. of shares', 'Price / share', and 'Withholding tax' columns.

    Returns:
    None

    Raises:
    ValueError: If the required columns are not present in the DataFrame.
    """
    required_columns = ['Action', 'Ticker', 'Time', 'No. of shares', 'Price / share', 'Withholding tax']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
    
    # Define dividend operations
    dividend_operations = [
        'Dividend (Ordinary)',
        'Dividend (Dividends paid by us corporations)',
        'Dividend (Dividend)',
        'Dividend (Tax exempted)'
    ]

    # Filter the DataFrame
    df_filtered = df[df['Action'].isin(dividend_operations)][['Ticker', 'Time', 'No. of shares', 'Price / share', 'Withholding tax']]
    
    # Ensure the 'Time' column is in datetime format
    df_filtered['Time'] = pd.to_datetime(df_filtered['Time'], errors='coerce')
    if df_filtered['Time'].isna().any():
        raise ValueError("Some dates in 'Time' column could not be converted to datetime.")
    
    # Get unique tickers
    div_tickers = df_filtered['Ticker'].unique().tolist()

    Div_Tax_Dict = {}

    for ticker in div_tickers:
        last_time = df_filtered[df_filtered['Ticker'] == ticker]['Time'].max()
        dff = df_filtered[(df_filtered['Ticker'] == ticker) & (df_filtered['Time'] == last_time)]
        
        try:
            n_shares = dff['No. of shares'].values[0]
            price = dff['Price / share'].values[0]
            tax = dff['Withholding tax'].values[0]
            if n_shares * price == 0:
                raise ValueError(f"Invalid value: No. of shares * Price / share is zero for ticker {ticker}.")
            tax_rate = tax / (n_shares * price)
        except Exception as e:
            raise ValueError(f"Error processing ticker '{ticker}': {e}")

        Div_Tax_Dict[ticker] = tax_rate

    # Save the dictionary to a JSON file
    try:
        with open('Div_Tax_Dict.json', 'w') as file:
            json.dump(Div_Tax_Dict, file, ensure_ascii=False, indent=4)
        print("Div tax dictionary has been updated and saved to 'Div_Tax_Dict.json'.")
    except Exception as e:
        raise ValueError(f"Error saving JSON file: {e}")
