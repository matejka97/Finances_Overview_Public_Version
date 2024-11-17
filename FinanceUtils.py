# This module contains all functions used in dividend class

import requests
import datetime
import pytz
import yfinance as yf
import pandas as pd
# ExchangeRate_API- Key
ExchangeRate_API_key = '7778a8f14b518ffa4133ab4e'

def exchange_rate_czk(foreign_currency):
    """
    Fetches the current exchange rate from the specified foreign currency to Czech Koruna (CZK) and
    returns it along with the date of the last update in datetime.datetime format.

    Parameters:
    foreign_currency (str): The currency code to fetch the exchange rate for.

    Returns:
    dict: A dictionary with keys 'date' (datetime.datetime), 'rate' (float), and 'currency' (str).
    """
    is_gbpence = foreign_currency == 'GBp'
    url = f'https://v6.exchangerate-api.com/v6/{ExchangeRate_API_key}/latest/{foreign_currency}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Try to parse the date, catch any errors to see the incorrect part
        try:
            date_format = "%a, %d %b %Y %H:%M:%S %z"
            last_update_date = datetime.datetime.strptime(data['time_last_update_utc'], date_format)
        except ValueError as e:
            print(f"Date parsing error: {e}")
            print(f"Received date string: {data['time_last_update_utc']}")
            raise
        rate = data['conversion_rates']['CZK']
        if is_gbpence:
            rate /= 100  # Convert GBP to GBp (pence to pounds)

        return {
            'date': last_update_date,
            'rate': rate,
            'currency': foreign_currency
        }
    else:
        response.raise_for_status()


def exchange_update(foreign_currency, original_exchange_rate_dict=None):
    """
    Updates or retrieves the stored exchange rate for a specified foreign currency, avoiding unnecessary API calls
    if the data is already up-to-date and matches the requested currency.

    Parameters:
    foreign_currency (str): The currency code to fetch or update the exchange rate for.
    Original_ExchangeRate_Dict (dict, optional): A dictionary containing the last fetched data with keys 'date', 'rate', and 'currency'.
        Defaults to None and initializes with minimal datetime, rate 1, and currency 'CZK'.

    Returns:
    dict: Updated or current exchange rate data dictionary.
    """

    # Initialize the dictionary if None is provided
    if original_exchange_rate_dict is None:
        original_exchange_rate_dict = {'date': datetime.datetime.min.isoformat(), 'rate': 1, 'currency': 'CZK'}

    # Collecting actual datetime from datetime module
    actual_date = datetime.datetime.now(pytz.utc)

    # Parse the date from the dictionary
    last_update_date = datetime.datetime.fromisoformat(original_exchange_rate_dict['date'])

    # Check if data needs updating based on currency and date
    if (foreign_currency != original_exchange_rate_dict['currency'] or
        last_update_date < actual_date - datetime.timedelta(hours=24)):
        # Data is outdated or for a different currency, need to update
        print("Updating exchange rate data...")
        exchange_rate_updated = exchange_rate_czk(foreign_currency)
        # Convert date to string for JSON serialization
        exchange_rate_updated['date'] = exchange_rate_updated['date'].isoformat()
    else:
        # Data is up-to-date and for the correct currency
        exchange_rate_updated = original_exchange_rate_dict
        print("Rate is actual and correct. No update needed.")

    return exchange_rate_updated

def collect_stock_info(ticker, shares):
    """
    Collects various financial metrics for a given stock ticker and calculates the total dividend for a specified number of shares.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
    shares (int): The number of shares held in the portfolio.

    Returns:
    tuple: A tuple containing the current stock price, dividend yield (as a percentage), CAGR of dividends, currency of the stock, and actual dividend income for the specified shares.

    Raises:
    ValueError: If the ticker does not have any dividend data or if the stock price is unavailable.
    Exception: For any other unexpected errors that might occur.
    """
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker)
        
        # Retrieve dividend history
        dividends = stock.dividends

        # Check if there are any dividends available
        if dividends.empty:
            raise ValueError(f"No dividend data available for ticker: {ticker}")
        
        # Retrieve stock split history
        splits = stock.splits
        # Adjust dividends for stock splits
        for split_date, ratio in splits.items():
            dividends.loc[dividends.index < split_date] /= ratio


        # Retrieve the current stock price (previous close)
        current_price = stock.info.get('previousClose')
        if current_price is None:
            raise ValueError(f"Stock price information is unavailable for ticker: {ticker}")

        # Dividend yield

        dividend_yield_perc = stock.info['dividendYield'] * 100

        # Convert dividends data to DataFrame
        dividends_df = dividends.to_frame().reset_index()
        dividends_df['Year'] = dividends_df['Date'].dt.year

        # Group by year and sum the dividends
        annual_dividends = dividends_df.groupby('Year')['Dividends'].sum().reset_index()

        # Calculate CAGR (Compound Annual Growth Rate)
        if len(annual_dividends) < 2:
            cagr_perc = None
            print(f"Not enough data to calculate CAGR for {ticker}")
        else:
            initial_dividend = annual_dividends['Dividends'].iloc[0]
            final_dividend = annual_dividends['Dividends'].iloc[-1]
            years = annual_dividends['Year'].iloc[-1] - annual_dividends['Year'].iloc[0]
            cagr = (final_dividend / initial_dividend) ** (1 / years) - 1
            cagr_perc = cagr * 100

        # Get data about currency
        currency = stock.info.get('currency')
        if currency is None:
            raise ValueError(f"Currency information is unavailable for ticker: {ticker}")

        # Calculate actual dividend of ticker in my portfolio
        actual_dividend = dividend_yield_perc/100 * shares * current_price

        return current_price, dividend_yield_perc, cagr_perc, currency, actual_dividend

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def update_stock_dict(input_dict, exchange_rates, div_tax_dict):
    """
    Updates stock information dictionary with calculated values in CZK.

    Parameters:
    input_dict (dict): Dictionary with stock tickers as keys and number of shares as values.
    exchange_rates (dict): Dictionary with currency exchange rates to CZK.
    div_tax_dict (dict): Dictionary with dividend tax rates for each ticker.

    Returns:
    dict: Updated dictionary with stock information including ticker, number of shares,
          stock value at close in CZK, annual dividend in CZK, dividend yield, stock price at close in CZK,
          stock currency, and stock tax on dividends.

    Raises:
    ValueError: If any of the input data is invalid.
    """
    required_keys = ['rate']
    for currency, data in exchange_rates.items():
        if not all(key in data for key in required_keys):
            raise ValueError(f"Exchange rates for '{currency}' must contain {required_keys} keys.")

    stock_dict = {
        'Ticker': [],
        'No.Shares': [],
        'Stock_Value_At_Close_CZK': [],
        'Year_Dividend_CZK': [],
        'Dividend_Yield': [],
        'Stock_Price_Close_CZK': [],
        'Stock_Currency': [],
        'Stock_Tax_Div': [],
        'Dividend_Growth': []
    }

    for ticker, shares in input_dict.items():
        try:
            # Get dividend, dividend yield, stock price, and currency
            #dividend = calculate_dividend(ticker, float(shares))
            #dividend_yield = calculate_dividend_yield(ticker)
            stock_price_at_close, dividend_yield_perc, cagr_perc, currency, actual_dividend = collect_stock_info(ticker, shares)

            # Check if the exchange rate for the currency is available
            if currency not in exchange_rates:
                raise ValueError(f"Exchange rate for currency '{currency}' not found.")

            exchange_rate = exchange_rates[currency]['rate']

            # Update stock dictionary
            stock_dict['Ticker'].append(ticker)
            stock_dict['No.Shares'].append(shares)
            stock_dict['Stock_Value_At_Close_CZK'].append(stock_price_at_close * shares * exchange_rate)
            stock_dict['Year_Dividend_CZK'].append(actual_dividend * exchange_rate)
            stock_dict['Dividend_Yield'].append(dividend_yield_perc)
            stock_dict['Stock_Price_Close_CZK'].append(stock_price_at_close * exchange_rate)
            stock_dict['Stock_Currency'].append(currency)
            stock_dict['Stock_Tax_Div'].append(div_tax_dict.get(ticker, 0))
            stock_dict['Dividend_Growth'].append(cagr_perc)

        except KeyError as e:
            print(f"Key error processing {ticker}: {e}")
        except ValueError as e:
            print(f"Value error processing {ticker}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {ticker}: {e}")

    return stock_dict
