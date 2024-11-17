import json
import pandas as pd
from FinanceUtils import update_stock_dict
from DataUpdate import total_deposit, trading_result, total_withdrawal, total_dividends, total_interest, market_sell_total, market_buy_total

with open('Stock_Dict.json','r') as file:
    input_dict = json.load(file)

with open('exchange_rates.json','r') as file:
    exchange_rates = json.load(file)

with open('Div_Tax_Dict.json', 'r') as file:
    div_tax_dict = json.load(file)



input_df = pd.read_csv(('origin.csv'))

start_date = pd.to_datetime(input_df["Time"]).min()
end_date = pd.to_datetime(input_df["Time"]).max()

deposit = total_deposit(input_df)
withdrawal = total_withdrawal(input_df)
dividends = total_dividends(input_df)
interest = total_interest(input_df)
result = trading_result(input_df, start_date, end_date)
sell = market_sell_total(input_df, start_date, end_date)
buy = market_buy_total(input_df, start_date, end_date)
free_funds = deposit + withdrawal + dividends + interest + sell - buy

print(f"Sell amount in 2024 {market_sell_total(input_df, "2024-01-01 00:00:00", "2024-12-31 00:00:00")}")
print(f'Deposit: {deposit} CZK')
print(f'Withdrawal: {withdrawal} CZK')
print(f'Total Dividend: {dividends} CZK')
print(f'Total gain from lending and cash interest: {interest}')
print(f'Gain/Loss from trading operations total: {result} CZK')
print(f'Free funds: {free_funds} CZK')


df = pd.DataFrame(update_stock_dict(input_dict, exchange_rates, div_tax_dict))

df_div = df[((df["Dividend_Yield"] > 2.5) & (df['Dividend_Yield'] < 20)) & ((df['Dividend_Growth'] > 5))]

invested = 66976.78
target_price = 480000
number_of_months = 240

n_div = len(df_div)

div_target = target_price/n_div

target_number_of_shares = div_target/df_div['Year_Dividend_CZK']

month_payments = pd.DataFrame({'Monthly_Payment_CZK':(target_number_of_shares * df_div['Stock_Price_Close_CZK'] + ( target_number_of_shares * df_div['Stock_Price_Close_CZK'] * df_div['Stock_Tax_Div']))/number_of_months})

df_div = pd.concat([df_div,month_payments], axis = 1)

print(f'Deposit monthly {df_div['Monthly_Payment_CZK'].sum()} CZK to have {target_price} CZK year dividend in {number_of_months/12} years.')
print('Monthly payment distribution:')
print(df_div[['Ticker','Monthly_Payment_CZK', 'Dividend_Yield', 'Dividend_Growth']])
print('Bad Dividend Growth titles - Sell When Green or At minimum lose:')
print(df[df['Dividend_Growth'] < 5][['Ticker','Dividend_Yield', 'Dividend_Growth']])
print(f'Actual Year Dividend {df['Year_Dividend_CZK'].sum()} CZK')
print(f'Actual value of whole portfolio: {df['Stock_Value_At_Close_CZK'].sum()} CZK')
print(f'Actual amount of whole money free funds included: {df['Stock_Value_At_Close_CZK'].sum() + free_funds} CZK')
print(f'Loss (-) or Gain (+) General: {df['Stock_Value_At_Close_CZK'].sum() + free_funds - deposit} CZK -> ' + 
      f'{((df['Stock_Value_At_Close_CZK'].sum() + free_funds - deposit)/deposit)*100} %')
print(f'Actual Loss (-) or Gain (+) Actual Portfolio: {df['Stock_Value_At_Close_CZK'].sum() - invested} CZK -> ' + 
      f'{((df['Stock_Value_At_Close_CZK'].sum() - invested)/invested)*100} %')

potential_titles = {"UU.L": 1,
                    "SDR.L": 1,
                    "SVT.L": 1,
                    "IGG.L": 1,
                    "RAT.L": 1,
                    "PETS.L": 1,
                    "MPE.L": 1,
                    "OCN.L": 1,
                    "CSN.L": 1,
                    "TEP.L": 1,
                    "ASHM.L": 1,
                    "POLR.L": 1,
                    "LIO.L": 1,
                    "RWS.L": 1,
                    "PHNX.L": 1,
                    "MNG.L": 1,
                    "BATS.L": 1,
                    "IMB.L": 1,
                    "PSN.L": 1,
                    "LGEN.L": 1,
                    "VOD.L": 1,
                    "MONY.L": 1,
                    "AV.L": 1,
                    "NG.L": 1,
                    "EON.DE": 1,
                    }

# df = pd.DataFrame(update_stock_dict(potential_titles, exchange_rates, div_tax_dict))
# df = df[['Ticker', "Dividend_Yield", "Dividend_Growth"]]
# df = df[((df["Dividend_Yield"] > 3) & (df['Dividend_Yield'] < 15)) & ((df['Dividend_Growth'] > 5) & (df['Dividend_Growth'] < 50))]
# print(df)