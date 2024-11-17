import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

pd.set_option('future.no_silent_downcasting', True)

def calculate_market_return(ticker, years=10):
    """
    Estimate the expected market return by calculating the average annual return 
    of a stock index over a given number of years.
    
    Parameters:
    ticker (str): The ticker symbol for the market index (e.g., '^GSPC' for S&P 500, '^FTSE' for FTSE 100)
    years (int): Number of years over which to calculate the average return
    
    Returns:
    float: Estimated average market return as a decimal
    """
    index = yf.Ticker(ticker)
    historical_data = index.history(period=f'{years}y')
    
    # Calculate yearly returns
    historical_data['Return'] = historical_data['Close'].pct_change().dropna()
    
    # Annualize the return by multiplying by 252 (assuming 252 trading days in a year)
    avg_annual_return = historical_data['Return'].mean() * 252
    return avg_annual_return

class Dividend:
    """
    Class for creating object for various dividend actions
    """
    def __init__(self, ticker_symbol, data_source=None):
        """
        Initialization
        """
        self.ticker_symbol = ticker_symbol
        self.stock = data_source or yf.Ticker(self.ticker_symbol)
        self.market_price = self.stock.info.get('previousClose', 0)
        self.error_messages = []
        self.dividends = None
        self.stock_splits = None
        self.dividends_split_corrected = None
        self.dividends_yearly = None
        self.dividends_split_corr_yearly = None
        self.div_growth_rate = None
        self.median_growth = None
        self.mad_growth = None
        self.normalized_mad = None
        self.fair_price = None
        self.dividend_current_year = None
        self.next_dividend = None
        self.avg_growth = None
        self.cost_of_equity = None
        self.market_return = None
        self.future_fcf = None
        self.market_cap = None
        self.total_debt = None
        self.tax_rate = None
        self.cost_of_debt = None
        self.wacc = None
        self.cagr_dividend = None
        self.intristic_value_per_share = None

        self.growth_volatility_NOT_checked = True
        self.suitability_for_DDM = False
        self.suitability_for_Multi_DDM = False
        self.suitability_for_DCF = False

        self.bond_yields = { 
                            'NYQ': 0.04240,  # 4.075% as a ratio
                            'NMS': 0.04240,
                            'LSE': 0.04235,  # 4.051%
                            'GER': 0.02291,  # 2.190%
                            'AMS': 0.02566   # 2.459%
                            }

    def get_market_return(self, market):
        """
        Initialize market return for the exchange if needed.
        """
        try:
            # Define default market returns if calculated market return is unexpectedly low
            default_market_returns = {
                'NYQ': 0.07,  # Default for NYQ (7%)
                'NMS': 0.07,  # Default for NMS (7%)
                'LSE': 0.065, # Default for LSE (6.5%)
                'GER': 0.065, # Default for GER (6.5%)
                'AMS': 0.065  # Default for AMS (6.5%)
                }

            # Get calculated market return for specific indices
            if market in ['NYQ', 'NMS']:
                calculated_return = calculate_market_return('^GSPC', years=10)
            elif market == 'LSE':
                calculated_return = calculate_market_return('^FTSE', years=10)
            elif market == 'GER':
                calculated_return = calculate_market_return('^GDAXI', years=10)
            elif market == 'AMS':
                calculated_return = calculate_market_return('^AEX', years=10)
            else:
                self.error_messages.append(f"get_market_return: WARNING - Unsupported market '{market}' for ticker {self.ticker_symbol}. Using default return of {default_market_returns.get(market, 0.07)}.")
                self.market_return = default_market_returns.get(market, 0.07)
                return

            # Validate the calculated return
            if calculated_return is None:
                self.error_messages.append(f"get_market_return: WARNING - Failed to calculate market return for market '{market}'. Using default return of {default_market_returns.get(market, 0.07)}.")
                self.market_return = default_market_returns.get(market, 0.07)
            else:
            # Check if market return is below the risk-free rate and set default if necessary

                rf = self.bond_yields.get(market, 0.02)  # Default risk-free rate

                if calculated_return > rf:
                    self.market_return = calculated_return
                else:
                    self.error_messages.append(f"get_market_return: WARNING - Calculated market return {calculated_return} for '{market}' is below or equal to risk-free rate {rf}. Using default return of {default_market_returns.get(market, 0.07)}.")
                    self.market_return = default_market_returns.get(market, 0.07)

        except Exception as e:
            print(f"An error in get_market_return occured for {self.ticker_symbol}. See error messages for details.")
            self.error_messages.append(f"get_market_return: ERROR - Exception in get_market_return for '{market}' on ticker '{self.ticker_symbol}': {e}")
            self.market_return = default_market_returns.get(market, 0.07)

    def fetch_dividend_data(self):
        """
        Collecting dividend data from yfinance
        """
        try:
            # Collect dividends
            self.dividends = self.stock.dividends

            # Check if any dividend data is available
            if self.dividends.empty:
                self.error_messages.append(f"fetch_dividend_data: ERROR - No dividend data found for ticker '{self.ticker_symbol}'.")
                self.dividends = None

        except Exception as e:
            print(f"An error in fetch_dividend_data occured for {self.ticker_symbol}. See error messages for details.")
            self.error_messages.append(f"fetch_dividend_data: ERROR - Failed to fetch dividend data for ticker '{self.ticker_symbol}': {e}")
    
    def correct_div_for_splits(self):
        """
        Correct dividend data for any stock splits in history
        """
        try: 
            # Collect stock.splits if pressent
            self.stock_splits = self.stock.splits

            # Ensure dividend data is available
            if self.dividends is None:
                self.fetch_dividend_data()

            # Check again if dividends data is None after attempting to fetch
            if self.dividends is None:
                self.error_messages.append(f"correct_div_for_splits: ERROR - No dividend data available for ticker '{self.ticker_symbol}' after fetching. Cannot apply split corrections.")
                self.dividends_split_corrected = None
                return

            # Check if stock splits data is available
            if self.stock_splits.empty:
                self.error_messages.append(f"correct_div_for_splits: INFO - No stock split data available for ticker '{self.ticker_symbol}'.")
                self.dividends_split_corrected = self.dividends
                return

            # Create a copy of dividends to apply adjustments
            dividends = self.dividends.copy()

            # Apply split adjustments
            for split_date, ratio in self.stock_splits.items():
                dividends.loc[dividends.index < split_date] /= ratio

            self.dividends_split_corrected = dividends
        except Exception as e:
            print(f"An error in correct_div_for_splits occurred for {self.ticker_symbol}. See error messages for details.")
            self.error_messages.append(f"correct_div_for_splits: ERROR - Exception occurred while adjusting dividends for ticker '{self.ticker_symbol}': {e}")
            self.dividends_split_corrected = None  # Set to None if correction fails
    
    def yearly_dividends(self):
        """
        Sum up yearly dividends.
        """
        try:
            # Ensure dividend data is available
            if self.dividends is None:
                self.fetch_dividend_data()
        
            # Check if dividends remain None after fetching
            if self.dividends is None:
                self.error_messages.append(f"yearly_dividends: ERROR - No dividend data available for ticker '{self.ticker_symbol}'. Cannot calculate yearly dividends.")
                self.dividends_yearly = None
                return

            # Group dividends by year and sum
            self.dividends_yearly = self.dividends.groupby(self.dividends.index.year).sum()
        
            # Ensure corrected dividends data is available for splits
            if self.dividends_split_corrected is None:
                self.correct_div_for_splits()
        
            # Check if dividends_split_corrected is None after correction attempt
            if self.dividends_split_corrected is None:
                self.error_messages.append(f"yearly_dividends: INFO - No corrected dividend data available for ticker '{self.ticker_symbol}'. Using uncorrected yearly dividends.")
                self.dividends_split_corr_yearly = self.dividends_yearly
                return

            # Sum yearly corrected dividends
            self.dividends_split_corr_yearly = self.dividends_split_corrected.groupby(
            self.dividends_split_corrected.index.year).sum()
        
        except Exception as e:
            print(f"An error occurred in yearly_dividends for {self.ticker_symbol}. See error messages for details.")
            self.error_messages.append(f"yearly_dividends: ERROR - Exception occurred while summing yearly dividends for '{self.ticker_symbol}': {e}")
            self.dividends_yearly = None
            self.dividends_split_corr_yearly = None  # Set to None if summing fails
    
    def get_growth_rate(self):
        """
        Calculate yearly dividend growth
        """
        try:
            # Ensure yearly dividends are calculated
            if self.dividends_split_corr_yearly is None:
                self.yearly_dividends()

            # Ensure there is enough data to calculate growth rate
            if self.dividends_split_corr_yearly is None or len(self.dividends_split_corr_yearly) < 2:
                self.error_messages.append(f"get_growth_rate: ERROR - Not enough data to calculate dividend growth for '{self.ticker_symbol}'.")
                self.div_growth_rate = None
                return

            # Calculate year-over-year growth rate
            growth_rates = (
                self.dividends_split_corr_yearly
                .pct_change()
                .replace([float('inf'), -float('inf')], None)
                .dropna()
            )

            # Check if growth rates are valid and assign
            if growth_rates.empty:
                self.error_messages.append(f"get_growth_rate: ERROR - Dividend growth rate calculation failed for '{self.ticker_symbol}'.")
                self.div_growth_rate = None
            else:
                self.div_growth_rate = growth_rates
            

            # Calculate CAGR if there’s enough data and no zero values at the start
            if len(self.dividends_split_corr_yearly) > 1:
                start_dividend = self.dividends_split_corr_yearly.iloc[0]
                end_dividend = self.dividends_split_corr_yearly.iloc[-1]
                years = len(self.dividends_split_corr_yearly) - 1

                if start_dividend > 0 and end_dividend > 0:
                    self.cagr_dividend = (end_dividend / start_dividend) ** (1 / years) - 1
                else:
                    self.error_messages.append(f"get_growth_rate: WARNING - Cannot calculate CAGR due to zero or negative starting dividend for '{self.ticker_symbol}'.")
                    self.cagr_dividend = None

        except Exception as e:
            print(f"An error occurred in get_growth_rate for {self.ticker_symbol}. See error messages for details.")
            self.error_messages.append(f"get_growth_rate: ERROR - Exception occurred while calculating growth rate for '{self.ticker_symbol}': {e}")
            self.div_growth_rate = None
            self.cagr_dividend = None  # Set to None if calculation fails
    
    def get_growth_rate_statistics(self):
        """
        Obtain base statistics from dividend growth rate data.
        """
        try:
            # Ensure dividend growth rate is available
            if self.div_growth_rate is None:
                self.get_growth_rate()

            # Proceed only if div_growth_rate has valid data
            if self.div_growth_rate is not None and not self.div_growth_rate.empty:
                # Remove outliers using the IQR method
                q1 = self.div_growth_rate.quantile(0.25)
                q3 = self.div_growth_rate.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered_growth = self.div_growth_rate[(self.div_growth_rate >= lower_bound) & (self.div_growth_rate <= upper_bound)]

                # Calculate statistics if filtered data is available
                if not filtered_growth.empty:
                    self.median_growth = filtered_growth.median()
                    self.avg_growth = filtered_growth.mean()
                    self.mad_growth = np.median(np.abs(filtered_growth - self.median_growth))
                
                    # Calculate normalized MAD, with handling if median_growth is zero
                    self.normalized_mad = self.mad_growth / self.median_growth if self.median_growth != 0 else float('inf')
                else:
                    self.error_messages.append(f"get_growth_rate_statistics: WARNING - No valid dividend growth data after filtering for '{self.ticker_symbol}'. Setting default statistics.")
                    self.median_growth = 0
                    self.avg_growth = 0
                    self.mad_growth = 0
                    self.normalized_mad = float('inf')
            else:
                # Default values if no dividend growth data is available
                self.error_messages.append(f"get_growth_rate_statistics: WARNING - No dividend growth data available for '{self.ticker_symbol}'. Setting default statistics.")
                self.median_growth = 0
                self.avg_growth = 0
                self.mad_growth = 0
                self.normalized_mad = float('inf')

        except Exception as e:
            print(f"An error occurred in get_growth_rate_statistics for {self.ticker_symbol}. See error messages for details.")
            self.error_messages.append(f"get_growth_rate_statistics: ERROR - Exception occurred while calculating growth statistics for '{self.ticker_symbol}': {e}")
            # Set defaults if an exception is encountered
            self.median_growth = 0
            self.avg_growth = 0
            self.mad_growth = 0
            self.normalized_mad = float('inf')

    def check_div_growth_volatility(self):
        """
        Check volatility of dividend growth rate to decide which model to use for fair share price estimation.
        """
        try:
            # Set flag to indicate volatility has been checked
            self.growth_volatility_NOT_checked = False

            # Ensure normalized MAD is available
            if self.normalized_mad is None:
                self.get_growth_rate_statistics()
        
            # Reset Suitability Flags
            self.suitability_for_DDM = False
            self.suitability_for_Multi_DDM = False
            self.suitability_for_DCF = False

            # Model selection based on normalized MAD value
            if self.normalized_mad is None or self.normalized_mad == float('inf'):
                self.error_messages.append(f"check_div_growth_volatility: WARNING - Unable to determine model suitability for '{self.ticker_symbol}' due to missing or unreliable volatility data.")
                return
        
            if self.normalized_mad <= 0.2:
                self.suitability_for_DDM = True
                self.error_messages.append(f"check_div_growth_volatility: INFO - '{self.ticker_symbol}' suitable for Dividend Discount Model (DDM) with normalized MAD {self.normalized_mad}.")
        
            elif 0.2 < self.normalized_mad <= 0.3:
                self.suitability_for_Multi_DDM = True
                self.error_messages.append(f"check_div_growth_volatility: INFO - '{self.ticker_symbol}' suitable for Multi-Stage DDM with normalized MAD {self.normalized_mad}.")
        
            else:
                self.suitability_for_DCF = True
                self.error_messages.append(f"check_div_growth_volatility: INFO - '{self.ticker_symbol}' suitable for Discounted Cash Flow Model (DCF) with normalized MAD {self.normalized_mad}.")
    
        except Exception as e:
            print(f"An error occurred in check_div_growth_volatility for {self.ticker_symbol}. See error messages for details.")
            self.error_messages.append(f"check_div_growth_volatility: ERROR - Exception occurred while determining model suitability for '{self.ticker_symbol}': {e}")
    
    def calculate_cost_of_equity(self):
        """
        Calculate cost of equity using the CAPM model.
        """
        try:
            # Retrieve the market exchange
            market = self.stock.info.get('exchange', None)
            if market is None:
                self.error_messages.append(f"calculate_cost_of_equity: ERROR - Exchange not found for ticker '{self.ticker_symbol}'.")
                self.cost_of_equity = None
                return

            # Get market return for the exchange
            self.get_market_return(market)

            # Retrieve risk-free rate and validate
            rf = self.bond_yields.get(market, 0.02)  # Default risk-free rate to 2% if missing
            if rf < 0:
                self.error_messages.append(f"calculate_cost_of_equity: WARNING - Invalid risk-free rate {rf} for market '{market}'. Using default of 2%.")
                rf = 0.02

            # Retrieve beta and validate
            beta = self.stock.info.get('beta', 1)  # Default beta to 1 if not available
            if beta < 0:
                self.error_messages.append(f"calculate_cost_of_equity: WARNING - Negative beta for '{self.ticker_symbol}'. Setting default beta of 1.")
                beta = 1

            # Validate and retrieve market return
            rm = self.market_return if self.market_return is not None else 0.07  # Default market return to 7% if missing
            if rm <= rf:
                self.error_messages.append(f"calculate_cost_of_equity: WARNING - Market return {rm} for '{market}' is less than or equal to risk-free rate {rf}. Using default equity premium.")
                rm = rf + 0.05  # Set to a 5% equity premium above rf if invalid

            # Calculate cost of equity using CAPM
            self.cost_of_equity = rf + beta * (rm - rf)
            self.error_messages.append(f"calculate_cost_of_equity: INFO - Cost of equity calculated for '{self.ticker_symbol}' as {self.cost_of_equity}.")

        except Exception as e:
            self.error_messages.append(f"calculate_cost_of_equity: ERROR - Exception in calculating cost of equity for '{self.ticker_symbol}': {e}")
            self.cost_of_equity = None  # Set to None if calculation fails"""

    def ddm_Calculate(self):
        """
        Calculate fair share price using the Dividend Discount Model (DDM).
        """
        try:
            # Ensure growth rate and cost of equity are available
            if self.div_growth_rate is None:
                self.get_growth_rate()
        
            if self.median_growth is None:
                self.get_growth_rate_statistics()
        
            if self.cost_of_equity is None:
                self.calculate_cost_of_equity()

            # Check that dividends for the current year are available
            current_year = datetime.now().year
            if self.dividends_split_corr_yearly is None or current_year not in self.dividends_split_corr_yearly:
                self.error_messages.append(f"ddm_Calculate: ERROR - No dividend data for the current year '{current_year}' for ticker '{self.ticker_symbol}'.")
                self.fair_price = None
                return None

            # Calculate next year's expected dividend
            self.dividend_current_year = self.dividends_split_corr_yearly[current_year]
            self.next_dividend = self.dividend_current_year * (1 + self.median_growth)

            # Ensure cost of equity exceeds growth rate
            if self.cost_of_equity is None or self.cost_of_equity <= self.median_growth:
                self.error_messages.append(f"ddm_Calculate: ERROR - Cost of equity {self.cost_of_equity} is less than or equal to growth rate {self.median_growth} for '{self.ticker_symbol}'. Fair price calculation is not feasible.")
                self.fair_price = None
                return None

            # Calculate fair price using the DDM formula
            self.fair_price = self.next_dividend / (self.cost_of_equity - self.median_growth)
            self.error_messages.append(f"ddm_Calculate: INFO - Calculated fair price for '{self.ticker_symbol}' as {self.fair_price}.")

            return self.fair_price

        except Exception as e:
            self.error_messages.append(f"ddm_Calculate: ERROR - Exception occurred while calculating fair price for '{self.ticker_symbol}': {e}")
            self.fair_price = None
            return None
    
    def future_fcf_estimation_easy(self, num_years=5):
        """
        Estimate future FCF using historical FCF data (easy method).
        """
        try:
            # Fetch and validate cash flow data
            cash_flow = self.stock.cashflow.fillna(0)
            if cash_flow.empty:
                self.error_messages.append(f"future_fcf_estimation_easy: WARNING - No cash flow data available for '{self.ticker_symbol}'.")
                self.future_fcf = None
                return

            # Extract historical Free Cash Flow and handle NaN values
            historical_fcf = cash_flow.loc['Free Cash Flow'][cash_flow.loc['Free Cash Flow'] != 0]
            if historical_fcf.isna().any():
                self.error_messages.append(f"future_fcf_estimation_easy: WARNING - NaN values in historical FCF for '{self.ticker_symbol}', using mean as fallback.")
                historical_fcf = historical_fcf.fillna(historical_fcf.mean())

            # Validate availability of historical FCF data
            if historical_fcf.empty:
                self.error_messages.append(f"future_fcf_estimation_easy: WARNING - No historical FCF data available for '{self.ticker_symbol}'.")
                self.future_fcf = None
                return

            # Calculate CAGR of FCF, handling zero or negative values
            fcf_start = historical_fcf.iloc[-1]  # Oldest FCF
            fcf_end = historical_fcf.iloc[0]  # Most recent FCF
            n = len(historical_fcf) - 1  # Years between start and end

            if fcf_start <= 0 or fcf_end <= 0:
                self.error_messages.append(f"future_fcf_estimation_easy: WARNING - Negative or zero FCF detected for '{self.ticker_symbol}', using conservative 1% growth.")
                cagr_fcf = 0.01  # Conservative growth rate of 1%
            else:
                cagr_fcf = (fcf_end / fcf_start)**(1 / n) - 1

            # Project future FCF and handle complex numbers
            initial_fcf = fcf_end
            self.future_fcf = [initial_fcf * (1 + cagr_fcf)**i for i in range(num_years)]
            self.future_fcf = [np.real(fcf) if np.iscomplex(fcf) else fcf for fcf in self.future_fcf]

            # Log completion of FCF estimation
            self.error_messages.append(f"future_fcf_estimation_easy: INFO - Future FCF estimated for '{self.ticker_symbol}' using CAGR of {cagr_fcf:.4f}.")

        except Exception as e:
            self.error_messages.append(f"future_fcf_estimation_easy: ERROR - Exception during FCF estimation for '{self.ticker_symbol}': {e}")
            self.future_fcf = None

    def future_fcf_estimation(self, num_years=5):
        """
        Estimate future FCF using a detailed method if data is available, 
        or fallback to the easy method if inputs are missing.
        """
        try:
            yearly_financials = self.stock.financials.fillna(0)
            cash_flow = self.stock.cashflow.fillna(0)
        
            # Check if critical financial data is available
            if yearly_financials.empty or cash_flow.empty:
                self.error_messages.append(f"future_fcf_estimation: WARNING - Missing financial data for '{self.ticker_symbol}'. Using easy method.")
                self.future_fcf_estimation_easy(num_years)
                return

            try:
                # Extract necessary financial metrics with fallback handling
                total_revenue = yearly_financials.loc['Total Revenue'][yearly_financials.loc['Total Revenue'] != 0]
                ebit = yearly_financials.loc['EBIT'][yearly_financials.loc['EBIT'] != 0]
                tax_rate_average = yearly_financials.loc['Tax Rate For Calcs'].mean()
                average_depreciation = yearly_financials.loc['Reconciled Depreciation'].mean()
            
                # Handle potential absence of CapEx data
                try:
                    capex = cash_flow.loc['Capital Expenditure']
                    average_capex = capex[capex < 0].mean()  # Only use negative CapEx values
                except KeyError:
                    self.error_messages.append(f"future_fcf_estimation: WARNING - Missing 'Capital Expenditure' data. Using easy method.")
                    self.future_fcf_estimation_easy(num_years)
                    return

                # Calculate working capital change ratio
                working_capital_change = cash_flow.loc['Change In Working Capital'][cash_flow.loc['Change In Working Capital'] != 0]
                historical_wc_to_revenue_ratio = (working_capital_change / total_revenue).mean()

                # Calculate EBIT margin and validate data
                ebit_margin = ebit / total_revenue
                revenue_start, revenue_end = total_revenue.iloc[0], total_revenue.iloc[-1]
                ebit_margin_start, ebit_margin_end = ebit_margin.iloc[0], ebit_margin.iloc[-1]
            
                # Ensure values are positive for CAGR calculations
                if revenue_start <= 0 or ebit_margin_start <= 0:
                    self.error_messages.append(f"future_fcf_estimation: WARNING - Negative or zero revenue or EBIT for '{self.ticker_symbol}'. Using easy method.")
                    self.future_fcf_estimation_easy(num_years)
                    return

                # Calculate CAGR
                n = len(total_revenue) - 1
                cagr_rev = (revenue_end / revenue_start) ** (1 / n) - 1
                cagr_ebit_marg = (ebit_margin_end / ebit_margin_start) ** (1 / n) - 1

                # Project future revenue and EBIT margin
                future_revenues = [revenue_end * (1 + cagr_rev) ** i for i in range(num_years)]
                future_ebit_margin = [ebit_margin_end * (1 + cagr_ebit_marg) ** i for i in range(num_years)]
                future_ebit = [revenue * margin for revenue, margin in zip(future_revenues, future_ebit_margin)]

                # Calculate EBIT after tax
                future_ebit_after_tax = [eb * (1 - tax_rate_average) for eb in future_ebit]

                # Forecast depreciation, CapEx, and working capital changes
                future_depreciation = [average_depreciation] * num_years
                future_capex = [average_capex] * num_years
                change_in_working_capital = [revenue * historical_wc_to_revenue_ratio for revenue in future_revenues]

                # Calculate FCF and handle NaNs
                self.future_fcf = [future_ebit_after_tax[i] + future_depreciation[i] - future_capex[i] - change_in_working_capital[i] for i in range(num_years)]
                self.future_fcf = [np.nan_to_num(fcf, nan=0.0) for fcf in self.future_fcf]

            except KeyError as e:
                self.error_messages.append(f"future_fcf_estimation: WARNING - Missing required data '{e}'. Using easy method.")
                self.future_fcf_estimation_easy(num_years)
                return

            # Log the successful calculation of future FCF
            self.error_messages.append(f"future_fcf_estimation: INFO - Future FCF estimated successfully for '{self.ticker_symbol}'.")

        except Exception as e:
            self.error_messages.append(f"future_fcf_estimation: ERROR - Exception occurred for '{self.ticker_symbol}': {e}")
            self.future_fcf = None

    def market_cap_estimate(self):
        """
        Estimate Market Cap based on stock price and shares outstanding.
        """
        try:
            # Fetch stock price and shares outstanding with defaults
            stock_price = self.market_price
            shares_outstanding = self.stock.info.get('sharesOutstanding', None)

            # Check if both stock price and shares outstanding are available
            if stock_price is None or shares_outstanding is None:
                self.error_messages.append(
                    f"market_cap_estimate: WARNING - Missing data for stock price or shares outstanding for '{self.ticker_symbol}'. Market cap not estimated."
                )
                self.market_cap = None
                return

            # Validate data integrity: positive and non-zero values
            if stock_price <= 0 or shares_outstanding <= 0:
                self.error_messages.append(
                    f"market_cap_estimate: WARNING - Invalid data for '{self.ticker_symbol}': stock price {stock_price} or shares outstanding {shares_outstanding} not positive. Market cap not estimated."
                )
                self.market_cap = None
            else:
                # Calculate market capitalization
                self.market_cap = stock_price * shares_outstanding
                self.error_messages.append(
                    f"market_cap_estimate: INFO - Market cap for '{self.ticker_symbol}' estimated successfully as {self.market_cap}."
                )

        except Exception as e:
            self.error_messages.append(f"market_cap_estimate: ERROR - Exception occurred for '{self.ticker_symbol}': {e}")
            self.market_cap = None
    
    def total_debt_estimate(self):
        """
        Estimate total debt by summing short-term and long-term debt.
        """
        try:
            # Initialize debt values as None to handle missing data effectively
            short_term_debt = 0
            long_term_debt = 0
        
            # Fetch the balance sheet and fill missing values with 0
            balance_sheet = self.stock.balance_sheet.fillna(0)
        
            # Retrieve 'Short Long Term Debt' if available
            if 'Short Long Term Debt' in balance_sheet.index:
                short_term_debt = balance_sheet.loc['Short Long Term Debt'].iloc[0]
            else:
                self.error_messages.append(
                    f"total_debt_estimate: WARNING - 'Short Long Term Debt' data not found for '{self.ticker_symbol}'. Assuming 0."
                )
        
            # Retrieve 'Long Term Debt' if available
            if 'Long Term Debt' in balance_sheet.index:
                long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[0]
            else:
                self.error_messages.append(
                    f"total_debt_estimate: WARNING - 'Long Term Debt' data not found for '{self.ticker_symbol}'. Assuming 0."
                )
        
            # Calculate total debt, ensuring we have valid values
            if short_term_debt < 0 or long_term_debt < 0:
                self.error_messages.append(
                    f"total_debt_estimate: WARNING - Negative debt values found for '{self.ticker_symbol}'. Setting total debt to None."
                )
                self.total_debt = None
            else:
                self.total_debt = short_term_debt + long_term_debt
                self.error_messages.append(
                    f"total_debt_estimate: INFO - Total debt for '{self.ticker_symbol}' estimated as {self.total_debt}."
                )

        except Exception as e:
            self.error_messages.append(
                f"total_debt_estimate: ERROR - Exception occurred for '{self.ticker_symbol}': {e}"
            )
            self.total_debt = None

    def tax_rate_estimate(self):
        """
        Estimate the effective tax rate of a company based on tax provision and pretax income.
        """
        try:
            # Fetch the income statement and handle missing data with .fillna(0)
            income_statement = self.stock.financials.fillna(0)
        
            # Initialize tax provision and pretax income
            tax_provision = income_statement.loc['Tax Provision'].iloc[0] if 'Tax Provision' in income_statement.index else None
            pretax_income = income_statement.loc['Pretax Income'].iloc[0] if 'Pretax Income' in income_statement.index else None
        
            # Check for missing values in both fields
            if tax_provision is None:
                self.error_messages.append(f"tax_rate_estimate: WARNING - 'Tax Provision' data not found for '{self.ticker_symbol}'.")
            if pretax_income is None:
                self.error_messages.append(f"tax_rate_estimate: WARNING - 'Pretax Income' data not found for '{self.ticker_symbol}'.")

            # Calculate the tax rate if pretax_income is valid and non-zero
            if pretax_income and pretax_income > 0:
                self.tax_rate = tax_provision / pretax_income
                self.error_messages.append(f"tax_rate_estimate: INFO - Tax rate for '{self.ticker_symbol}' estimated as {self.tax_rate:.2%}.")
            else:
            # Set tax_rate to None if pretax_income is zero or missing
                self.tax_rate = None
                self.error_messages.append(f"tax_rate_estimate: WARNING - Insufficient data to calculate tax rate for '{self.ticker_symbol}'. Defaulting to None.")

        except Exception as e:
            # Catch exceptions and log the error message
            self.error_messages.append(f"tax_rate_estimate: ERROR - Exception occurred for '{self.ticker_symbol}': {e}")
            self.tax_rate = None

    def cost_of_debt_estimate(self):
        """
        Estimate the cost of debt based on interest expense and total debt, with error handling.
        """
        try:
            # Retrieve interest expense, setting it to None if missing
            interest_expense = self.stock.financials.loc['Interest Expense'].iloc[0] if 'Interest Expense' in self.stock.financials.index else None

            # Log a warning if interest expense is missing
            if interest_expense is None:
                self.error_messages.append(f"cost_of_debt_estimate: WARNING - 'Interest Expense' not found for ticker '{self.ticker_symbol}'. Setting cost_of_debt to None.")
                self.cost_of_debt = None
                return

            # Estimate total debt if not already set
            if self.total_debt is None:
                self.total_debt_estimate()

            # Calculate cost of debt if total debt is positive
            if self.total_debt and self.total_debt > 0:
                self.cost_of_debt = abs(interest_expense / self.total_debt)
                self.error_messages.append(f"cost_of_debt_estimate: INFO - Cost of debt estimated as {self.cost_of_debt:.2%} for ticker '{self.ticker_symbol}'.")
            else:
                self.error_messages.append(f"cost_of_debt_estimate: WARNING - Total debt is zero or undefined for '{self.ticker_symbol}'. Setting cost_of_debt to None.")
                self.cost_of_debt = None

        except Exception as e:
            # Log any exceptions encountered during calculation
            self.error_messages.append(f"cost_of_debt_estimate: ERROR - Exception occurred for '{self.ticker_symbol}': {e}")
            self.cost_of_debt = None
    
    def calculate_wacc(self):
        """
        Calculate the Weighted Average Cost of Capital (WACC)
        """
        try:
            # Ensure all necessary components are estimated
            if self.market_cap is None:
                self.market_cap_estimate()
    
            if self.cost_of_equity is None:
                self.calculate_cost_of_equity()

            if self.cost_of_debt is None:
                self.cost_of_debt_estimate()
    
            if self.tax_rate is None:
                self.tax_rate_estimate()

            # Assign values, defaulting to zero if data is missing
            E = self.market_cap if self.market_cap is not None else 0
            D = self.total_debt if self.total_debt is not None else 0
            V = E + D  # Total firm value

            # Handle case where there's no debt
            if D == 0:
                self.error_messages.append(f"calculate_wacc: INFO - No debt for ticker '{self.ticker_symbol}', WACC is set to cost of equity.")
                self.wacc = self.cost_of_equity
                return

            # Check if V is zero to prevent division by zero
            if V == 0:
                self.error_messages.append(f"calculate_wacc: ERROR - Total firm value (E + D) is zero for '{self.ticker_symbol}', cannot calculate WACC.")
                self.wacc = None
                return

            # Ensure cost of equity and cost of debt are available for the calculation
            r_e = self.cost_of_equity
            r_d = self.cost_of_debt
            T = self.tax_rate if self.tax_rate is not None else 0

            if r_e is None or r_d is None:
                self.error_messages.append(f"calculate_wacc: ERROR - Missing cost of equity or cost of debt for '{self.ticker_symbol}', cannot calculate WACC.")
                self.wacc = None
                return

            # Calculate WACC
            self.wacc = ((E / V) * r_e) + ((D / V) * r_d * (1 - T))
            self.error_messages.append(f"calculate_wacc: INFO - WACC calculated successfully for '{self.ticker_symbol}' as {self.wacc:.2%}.")

        except Exception as e:
            self.error_messages.append(f"calculate_wacc: ERROR - Exception occurred for '{self.ticker_symbol}': {e}")
            self.wacc = None  # Set WACC to None if an error occurs

    def show_error_messages(self):
        """
        Prints error, warning, and info messages.
        """
        if not self.error_messages:
            print('No Errors/Warnings/Infos')
        else:
            print("Messages Log:")
            print("=" * 30)  # Separator for readability
            for message in self.error_messages:
                print(message)
    
    def calculate_intrinsic_value(self, perpetual_growth_rate=0.02):
        """
        Calculate the intrinsic value per share using the Discounted Cash Flow (DCF) model.
        """
        try:
            # Ensure future FCFs and WACC are available
            if self.future_fcf is None:
                self.future_fcf_estimation()
        
            if self.wacc is None:
                self.calculate_wacc()

            # Validate WACC against perpetual growth rate to prevent invalid terminal value calculations
            if self.wacc <= perpetual_growth_rate:
                self.error_messages.append(f"calculate_intrinsic_value: ERROR - WACC ({self.wacc}) must be greater than perpetual growth rate ({perpetual_growth_rate}).")
                self.intrinsic_value_per_share = None
                return

            # Step 1: Calculate Present Value of Projected FCFs
            discounted_fcf = [
                fcf / (1 + self.wacc) ** year for year, fcf in enumerate(self.future_fcf, start=1)
            ]
    
            # Step 2: Calculate Terminal Value and Discount it to Present Value
            terminal_value = (self.future_fcf[-1] * (1 + perpetual_growth_rate)) / (self.wacc - perpetual_growth_rate)
            discounted_terminal_value = terminal_value / (1 + self.wacc) ** len(self.future_fcf)

            # Step 3: Sum of Present Values
            intrinsic_value = sum(discounted_fcf) + discounted_terminal_value

            # Step 4: Intrinsic Value per Share
            shares_outstanding = self.stock.info.get('sharesOutstanding', None)
            if shares_outstanding is None or shares_outstanding <= 0:
                self.error_messages.append(f"calculate_intrinsic_value: ERROR - Invalid or missing 'sharesOutstanding' for {self.ticker_symbol}. Cannot calculate intrinsic value per share.")
                self.intrinsic_value_per_share = None
                return

            self.intrinsic_value_per_share = intrinsic_value / shares_outstanding

        except ZeroDivisionError:
            self.error_messages.append(f"calculate_intrinsic_value: ERROR - Division by zero encountered in intrinsic value calculation for {self.ticker_symbol}.")
            self.intrinsic_value_per_share = None
    
        except Exception as e:
            self.error_messages.append(f"calculate_intrinsic_value: ERROR - Exception occurred for {self.ticker_symbol}: {e}")
            self.intrinsic_value_per_share = None


    def fair_share_price(self):
        """
        Calculate fair share price according to the most suitable valuation model (DDM or DCF).
        """
        try:
            # Check growth volatility if not already checked
            if self.growth_volatility_NOT_checked:
                self.check_div_growth_volatility()

            # Check if DDM is suitable and calculate fair price
            if self.suitability_for_DDM:
                self.fair_price = self.ddm_Calculate()

                # Ensure the DDM calculation succeeded
                if self.fair_price is None:
                    self.error_messages.append(f"fair_share_price: WARNING - DDM calculation failed or was unsuitable for {self.ticker_symbol}. Falling back to intrinsic value calculation.")
                    self.calculate_intrinsic_value()
                    self.fair_price = self.intrinsic_value_per_share
            else:
                # Calculate intrinsic value using DCF as fallback
                self.calculate_intrinsic_value()
                self.fair_price = self.intrinsic_value_per_share

            # Final validation of the fair price
            if self.fair_price is None:
                self.error_messages.append(f"fair_share_price: ERROR - Both DDM and DCF calculations failed for {self.ticker_symbol}. Fair price could not be determined.")
        except Exception as e:
            self.error_messages.append(f"fair_share_price: ERROR - Exception occurred while calculating fair price for {self.ticker_symbol}: {e}")
            self.fair_price = None
    
    def calculate_potential_gain_loss(self):
        """
        Evaluate potential gain or loss based on the difference between fair price and market price.
        """
        try:
            # Ensure both fair price and market price are available
            if self.fair_price is not None and self.market_price is not None:
                # Check if market price is positive to avoid division by zero or negative values
                if self.market_price > 0:
                    # Calculate the gain or loss percentage
                    gain_loss_percentage = ((self.fair_price - self.market_price) / self.market_price) * 100

                    # Display the potential gain or loss
                    if gain_loss_percentage > 0:
                        print(f"Potential Gain: {gain_loss_percentage:.2f}%")
                    elif gain_loss_percentage < 0:
                        print(f"Potential Loss: {gain_loss_percentage:.2f}%")
                    else:
                        print("No potential gain or loss, fair price equals market price.")
                
                    # Return the percentage for further use
                    return gain_loss_percentage
                else:
                    print("Invalid market price (non-positive). Cannot calculate potential gain/loss.")
                    return None
            else:
                print("Fair price or market price is missing. Cannot calculate potential gain/loss.")
                return None
        except Exception as e:
            print(f"Error calculating potential gain/loss for {self.ticker_symbol}: {e}")
            return None