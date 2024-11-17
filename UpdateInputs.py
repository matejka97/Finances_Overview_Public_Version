import json
import pandas as pd
from FinanceUtils import exchange_update
from DataUpdate import update_main_csv, update_stock_dict, update_tax_dict

main_file = 'origin.csv'
source_files = 'data_trading212'

update_main_csv(main_file, source_files)

df = pd.read_csv('origin.csv')

update_stock_dict(df)

update_tax_dict(df)

currencies  = ['USD','EUR','GBp']

with open('exchange_rates.json','r') as file:
    exchange_rates = json.load(file)

for currency in currencies:
    exchange_rates[currency] = exchange_update(currency)

with open('exchange_rates.json','w') as file:
    # Save dictionary into JSON
    json.dump(exchange_rates, file, ensure_ascii = False, indent = 4)