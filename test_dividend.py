import pandas as pd
import numpy as np
import unittest
from unittest.mock import MagicMock
from datetime import datetime
from Dividends import Dividend

# Creating a comprehensive mock dataset

# Mocking self.stock.info data
mock_info = {
    'previousClose': 150.0,
    'beta': 1.2,
    'sharesOutstanding': 5000000,  # Example share count
    'exchange': 'NYQ'
}

# Mocking self.stock.dividends data as a DataFrame
dividend_data = {
    'Date': [
        '2018-01-01', '2018-04-01', '2018-07-01', '2018-10-01',
        '2019-01-01', '2019-04-01', '2019-07-01', '2019-10-01',
        '2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01',
        '2021-01-01', '2021-04-01', '2021-07-01', '2021-10-01'
    ],
    'Dividends': [
        0.5, 0.5, 0.5, 0.5,  # 2018
        0.6, 0.6, 0.6, 0.6,  # 2019
        0.65, 0.65, 0.65, 0.65,  # 2020
        0.7, 0.7, 0.7, 0.7   # 2021
    ]
}
mock_dividends = pd.DataFrame(dividend_data)
mock_dividends['Date'] = pd.to_datetime(mock_dividends['Date'])
mock_dividends.set_index('Date', inplace=True)

# Mocking self.stock.splits data as a Series
split_data = {'2020-07-01': 2}  # A 2-for-1 stock split on July 1, 2020
mock_splits = pd.Series(split_data)
mock_splits.index = pd.to_datetime(mock_splits.index)

# Mocking self.stock.cashflow data as a DataFrame
cashflow_data = {
    'Free Cash Flow': [-200000, 500000, 550000, 600000],  # Yearly FCF data
}
mock_cashflow = pd.DataFrame(cashflow_data)
mock_cashflow.index = pd.to_datetime(['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31'])

# Mocking self.stock.financials data as a DataFrame
financials_data = {
    'Total Revenue': [1000000, 1100000, 1150000, 1200000],
    'EBIT': [200000, 220000, 230000, 240000],
    'Tax Provision': [30000, 35000, 38000, 40000],
    'Pretax Income': [250000, 255000, 260000, 265000],
    'Reconciled Depreciation': [10000, 12000, 15000, 20000],
    'Capital Expenditure': [-50000, -60000, -55000, -58000]
}
mock_financials = pd.DataFrame(financials_data)
mock_financials.index = pd.to_datetime(['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31'])

# Mocking self.stock.balance_sheet data as a DataFrame
balance_sheet_data = {
    'Short Long Term Debt': [150000, 160000, 170000, 180000],
    'Long Term Debt': [500000, 510000, 520000, 530000]
}
mock_balance_sheet = pd.DataFrame(balance_sheet_data)
mock_balance_sheet.index = pd.to_datetime(['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31'])

# Now, create the mock data source
mock_data_source = MagicMock()
mock_data_source.info = mock_info
mock_data_source.dividends = mock_dividends
mock_data_source.splits = mock_splits
mock_data_source.cashflow = mock_cashflow
mock_data_source.financials = mock_financials
mock_data_source.balance_sheet = mock_balance_sheet

class TestDividend(unittest.TestCase):

    def setUp(self):
        # Set up comprehensive mock data source for testing
        self.mock_data_source = MagicMock()
        self.mock_data_source.info = {
            'previousClose': 150.0,
            'beta': 1.2,
            'sharesOutstanding': 5000000,
            'exchange': 'NYQ'
        }
        
        dividend_data = {
            'Date': ['2018-01-01', '2018-04-01', '2019-01-01', '2019-04-01', '2020-01-01', '2021-01-01'],
            'Dividends': [0.5, 0.5, 0.6, 0.6, 0.65, 0.7]
        }
        mock_dividends = pd.DataFrame(dividend_data)
        mock_dividends['Date'] = pd.to_datetime(mock_dividends['Date'])
        mock_dividends.set_index('Date', inplace=True)
        self.mock_data_source.dividends = mock_dividends

        # Initialize Dividend instance with mock data source
        self.div = Dividend("AAPL", data_source=self.mock_data_source)

    def test_calculate_cost_of_equity(self):
        self.div.calculate_cost_of_equity()
        self.assertAlmostEqual(self.div.cost_of_equity, 0.08, places=2)

    def test_market_cap_estimate(self):
        self.div.market_cap_estimate()
        self.assertEqual(self.div.market_cap, 750000000)

    def test_yearly_dividends(self):
        self.div.fetch_dividend_data()
        self.div.yearly_dividends()
        expected_yearly_dividends = pd.Series({2018: 1.0, 2019: 1.2, 2020: 1.3, 2021: 1.4})
        pd.testing.assert_series_equal(self.div.dividends_yearly, expected_yearly_dividends)

    def test_get_growth_rate(self):
        self.div.fetch_dividend_data()
        self.div.yearly_dividends()
        self.div.get_growth_rate()
        expected_growth_rate = pd.Series({2019: 0.2, 2020: 0.0833, 2021: 0.0769})
        pd.testing.assert_series_equal(self.div.div_growth_rate, expected_growth_rate)

if __name__ == "__main__":
    unittest.main()