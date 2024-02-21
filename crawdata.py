import csv
from binance.client import Client
import pandas as pd
from csv import writer
#symbol sẽ là 1 cái list gồm 100 mã cổ phiếu của binance
SYMBOL = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','1INCHUSDT',
          'ALCXUSDT','ALGOUSDT','ALICEUSDT','ALPACAUSDT','ALPHAUSDT',
            'AAVEUSDT','ACAUSDT','ACMUSDT','ALPINEUSDT','ACHUSDT',
            'ADADOWNUSDT', 'ADAUPUSDT','ADAUSDT','ADXUSDT','AERGOUSDT',
           'AEURUSDT','AGIXUSDT','AGLDUSDT','AIONUSDT','AIUSDT',
           'AKROUSDT','ADATRY','ADATUSD','ADAUPUSDT','ADAUSDC',
          'ADAUSDT','AXSBUSD','ADXBTC','ADXETH','ATOMUSDC',
          'ATOMUSDT','AUCTIONBTC','AERGOBTC','AERGOBUSD','AUCTIONBUSD',
          'AUDBUSD','AUDIOBTC','AGIXBTC','AUDIOBUSD','AIONBTC',
          'AUDIOUSDT','AIONETH','AIONUSDT','AKROBTC','AKROUSDT',
          'ALGOBNB','ALGOBTC','ALGOBUSD','AUDUSDT','AUTOBTC',
          'ALGOUSDT','ALICEBTC','ALICEBUSD','ALICEUSDT','ALPHABNB',
          'ALPHABTC','ALPHABUSD','ALPHAUSDT','AUTOBUSD','AMBBTC',
          'AUTOUSDT','ANKRBNB','ANKRBTC','AVABNB','AVABTC',
          'AVABUSD','ANKRUSDT','ANTBNB','ANTBTC','ANTBUSD',
          'ANTUSDT','AVAUSDT','APPCBTC','AVAXBNB','ARBNB',
          'ARBTC','ARBUSD','AVAXBTC','ARDRBTC','AVAXBUSD',
          'ARDRUSDT','ARKBTC','AVAXEUR','AVAXTRY','AVAXUSDT',
          'ASRUSDT','ASTBTC','AXSBNB','ATABNB','ATABTC',
          'ATABUSD','ATAUSDT','ATMBTC','ATMBUSD','ATMUSDT']
n = int(input("Enter the number of symbols to crawl: "))

# Select the first n symbols
selected_symbols = SYMBOL[:n]
# đến ALTUSDT rồi nha
# No API key/secret needed for this type of call
client = Client()

columns = [ 
    'Asset','Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'ignore'
]
#đây sẽ là 1 vòng lặp để lấy từng symbol trong 100 mã cổ phiếu đó đánh số từ 0 đến 99
# sothutucophieu = 0
# while sothutucophieu < 2:
#     sothutucophieu = sothutucophieu + 1
filename = f'{n}assetraw.csv'

with open(filename, 'w') as f:
    write = writer(f)
    write.writerow(columns)
    
    for symbol in selected_symbols :
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "30 April, 2023", "30 June, 2023")
        #print(f"The symbol {symbol} has {len(klines)} rows.")  # Print the number of rows
        for kline in klines:
            write.writerow([symbol] + kline)

#df['#'] = y  
#df.to_csv('BTCUSDT.csv')  
# cần 1 file để tổng hợp lại tất cả symbol, moi lan them la cu append cai cu vao, nay la append 1 dong

