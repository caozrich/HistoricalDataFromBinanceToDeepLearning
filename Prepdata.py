from getDatafromBinance import getOHLCfromPair
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM ,Dropout, BatchNormalization
from ta.trend import SMAIndicator
from binance.client import Client

API_KEY = "Your API key"
API_SECRET = "API_SECRET"

client = Client(API_KEY, API_SECRET, tld="com")



class MainStrategy():
    
    def __init__(self) :

        self.getDataFromBinance()

    def getDataFromBinance(self):
    
        realdata      =   getOHLCfromPair(client,"BTCUSDT")
        datareal      =   self.indicator(realdata)
        x            =   self.preprocesingData(datareal) 

        self.train_algorthm(x,12,2) #slice of train data for do predictions and hours for forescast
        

    def indicator(self,df): #some libs like TA needs a df with datetime as index
        
        raw = df.copy()
        raw.reset_index(drop=True, inplace=True)
        raw['Date']   = raw['Gmt time']
        raw           = raw.drop('Gmt time', axis=1)

        raw['Date']        = pd.to_datetime(raw['Date'],dayfirst=True)

        raw.set_index('Date',inplace=True)

        raw.columns         = [i.lower() for i in raw.columns]
        
        raw['SMA'] = SMAIndicator(raw['close'],7, True).sma_indicator()   
        
        return raw


        
    def timeSeriestoSupervised(self,X,timesteps,n_target):
        
    
        x = np.zeros( [len(X)-(timesteps+n_target), timesteps, X.shape[1] ])
        y = np.zeros( [len(X)-(timesteps+n_target), n_target ])   

        for t in range(timesteps):
            x[:,t] = X[t:-(timesteps+n_target)+t,:]
        for i in range(n_target):
            y[:,i] = X[timesteps+i:-(n_target-i),0]

        return x,y    


    

    def train_algorthm(self,X,timesteps,n_target):
    
        x,y = self.timeSeriestoSupervised(X,timesteps,n_target)


        print(x.shape)
        print(x[-1,:,0])
        

        precio_ultima_hora = x[:,-1,0]
        precio_dos_horas_despues = y[:,1]
        target= precio_dos_horas_despues-precio_ultima_hora
        target[target<=0] = 0
        target[target>0] = 1
        print(target.shape)
        print(target[:10])


        from sklearn.utils import shuffle
        X,target = shuffle(x,target)

        tt=int(0.9*len(X))

        x_train,x_test = X[:tt],X[tt:]
        y_train,y_test = target[:tt],target[tt:]

        y_train = y_train.reshape(y_train.shape[0],1)
        y_test = y_test.reshape(y_test.shape[0],1)


  
        model = Sequential()
        
        model.add(LSTM(128, input_shape=( x_train.shape[1], x_train.shape[2] ), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))


        model.add(LSTM(128, input_shape=( x_train.shape[1], x_train.shape[2] ), return_sequences=False))

        model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
        model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
        
        model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

        print(model.summary ())

        model.fit(x_train,y_train,batch_size=64, epochs=128) 





    def preprocesingData(self,data):
        
        data = data.drop(data.index.values[:45])

        data = data[['close','high','low','SMA']]
        
        data.reset_index(level=0, inplace=True)
        data.drop('Date', axis=1, inplace=True)


        x = data.values #df to numpy array

        
        scaler0 = StandardScaler()
        scaler0.fit(x[:,0].reshape(x[:,0].shape[0],1)) 
        x[:,0] = (scaler0.transform(x[:,0].reshape(x[:,0].shape[0],1))).flatten()
  
        scaler1 = StandardScaler() 
        scaler1.fit(x[:,1].reshape(x[:,1].shape[0],1))
        x[:,1] = (scaler1.transform(x[:,1].reshape(x[:,1].shape[0],1))).flatten()        

        scaler2 = StandardScaler() 
        scaler2.fit(x[:,2].reshape(x[:,2].shape[0],1))
        x[:,2] = (scaler2.transform(x[:,2].reshape(x[:,2].shape[0],1))).flatten()    
                
        scaler3 = StandardScaler() 
        scaler3.fit(x[:,3].reshape(x[:,3].shape[0],1))
        x[:,3] = (scaler3.transform(x[:,3].reshape(x[:,3].shape[0],1))).flatten()     
        
        return x   
        
        
tester = MainStrategy()        
