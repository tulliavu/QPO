import csv
from binance.client import Client
import pandas as pd
from csv import writer

SYMBOL = 'LOOMBTC'

# No API key/secret needed for this type of call
client = Client()

columns = [ 
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'ignore'
]

klines = client.get_historical_klines(SYMBOL, Client.KLINE_INTERVAL_1DAY, "21 Oct, 2021", "21 Oct, 2022")
i = 0
y = []
while i < 366:
    i = i+1
    y.append(i)
with open('LOOMBTC.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(columns)
    write.writerows(klines)
    
df = pd.read_csv('LOOMBTC.csv')
#df['#'] = y  
#df.to_csv('BTCUSDT.csv')  
