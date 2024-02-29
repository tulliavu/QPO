import csv
from binance.client import Client
import pandas as pd
from csv import writer
#symbol sẽ là 1 cái list gồm 100 mã cổ phiếu của binance
SYMBOL = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','1INCHUSDT',
          'ALCXUSDT','ALGOUSDT','ALICEUSDT','ALPACAUSDT','ALPHAUSDT',
            'AAVEUSDT','ACAUSDT','ACMUSDT','ALPINEUSDT','ACHUSDT',
            'ADADOWNUSDT', 'ADAUPUSDT','ADAUSDT','ADXUSDT','BUSDUSDT',
           'CFXUSDT','AGIXUSDT','AGLDUSDT','C98USDT','CAKEUSDT',
           'AKROUSDT','CELOUSDT','AMBUSDT','AMPUSDT','CELRUSDT',
          'ANKRUSDT','ANTUSDT','CHESSUSDT','APEUSDT','API3USDT',
          'APTUSDT','ARBUSDT','ARDRUSDT','CHRUSDT','CHZUSDT',
          'BURGERUSDT','ARPAUSDT','ARUSDT','ASRUSDT','ASTRUSDT',
          'CITYUSDT','ATAUSDT','ATMUSDT','ATOMUSDT','AUCTIONUSDT',
          'AUDIOUSDT','CKBUSDT','CLVUSDT','AVAUSDT','AVAXUSDT',
          'AXSUSDT','BADGERUSDT','BAKEUSDT','BALUSDT','BANDUSDT',
          'BARUSDT','BATUSDT', 'CTKUSDT','CTSIUSDT','COMPUSDT',
          'COSUSDT','COTIUSDT','CTXCUSDT','DATAUSDT','CVPUSDT',
          'AVABUSD','CRVUSDT','CVXUSDT','BELUSDT','BETAUSDT',
          'BICOUSDT','BIFIUSDT','DCRUSDT','DEGOUSDT','BLZUSDT',
          'DARUSDT','DASHUSDT','BNBDOWNUSDT','BNBUPUSDT','BNBUSDT',
          'BNTUSDT','BNXUSDT','BONDUSDT','DENTUSDT',
          'BSWUSDT','DFUSDT','DGBUSDT','BTCUPUSDT','BTCUSDT',
          'DIAUSDT','BTSUSDT','BTTCUSDT','DODOUSDT','DOCKUSDT']
n = int(input("Enter the number of symbols to crawl: "))

# Select the first n symbols
selected_symbols = SYMBOL[:n]
# đến BUSDUSDT rồi nha
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

