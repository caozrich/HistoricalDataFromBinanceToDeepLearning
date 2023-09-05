from datetime import datetime,timedelta
import pytz
import datetime as dt
import time
import pandas as pd



def getOHLCfromPair(client,symbol):

    timeEnd = datetime.now(tz = pytz.timezone('Etc/GMT-5'))
    delta = timedelta(hours = 1)

    timeStart = timeEnd - (300*delta) #Some indicators require extra candlesticks for calculations

    timeStart = timeStart.isoformat()
    klines = None

    interval="1h" 
    client.KLINE_INTERVAL_1HOUR

    try :
        klines = client.get_historical_klines(symbol, interval, timeStart)

    except Exception as e:
        print(e)
          
        time.sleep(5)
        klines = client.get_historical_klines(symbol, interval, timeStart)
        

    
    data = pd.DataFrame(klines)
    # create colums name
    data.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
                
    # change the timestamp 
    data['Gmt time'] = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]

    df=data[['open', 'high', 'low', 'close']].astype(float)
    df.rename(columns = {'open':'Open','high':'High', 'low':'Low','close':'Close',}, inplace = True)
    df['Gmt time'] =  pd.to_datetime(data["Gmt time"], unit='s').apply(lambda x: x.replace(minute = 0,second=0,microsecond=0)) #datetime format correction 
    df = df[['Gmt time','Open', 'High', 'Low', 'Close']]  


    return df

# NOTE: If you don't have a Binance API key or prefer not to use one, you can use the following code:
def get_binance_bars(symbol, interval, startTime, endTime):
    
    url = "https://data.binance.com/api/v3/klines"
    
    # print(endTime)
 
    startTime = str(int(startTime.timestamp() * 1000))
    endTime = str(int(endTime.timestamp() * 1000))
    limit = '1000'
 
    req_params = {"symbol" : symbol, 'interval' : interval, 'startTime' : startTime, 'endTime' : endTime, 'limit' : limit}
 
    df = pd.DataFrame(json.loads(requests.get(url, params = req_params).text))
 
    if (len(df.index) == 0):
        return None
     
    df = df.iloc[:, 0:6]
    df.columns = ['Gmt time', 'open', 'high', 'low', 'close', 'volume']
    
 
    df.open      = df.open.astype("float")
    df.high      = df.high.astype("float")
    df.low       = df.low.astype("float")
    df.close     = df.close.astype("float")
    df.volume    = df.volume.astype("float")
    
    df['adj_close'] = df['close']

    return df
