import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
import pyrenn as prn

def save_to_csv_from_yahoo(ticker, syear, smonth, sday, eyear, emonth, eday):
    start = dt.datetime(syear, smonth, sday)
    end = dt.datetime(eyear, emonth, eday)
    
    df = web.DataReader(ticker, 'yahoo', start, end)
    
    df.to_csv("/Users/PCComp/Documents/TFG/data/" + ticker + '.csv')
    return df

def get_df_from_csv(ticker):
    try:
        df = pd.read_csv("/Users/PCComp/Documents/TFG/data/" + ticker + '.csv')
    except FileNotFoundError:
        print("File Not Found")
    else:
        return df

df = save_to_csv_from_yahoo('AMZN', 2020, 1, 1, 2020, 12, 30)

P = df['Open'].values
Y = df['Close'].values
Ptest_ = df['Open'].values
Ytest_ = df['Adj Close'].values

# Use the timesteps t = [3..99] as Test Data
Ptest = Ptest_[3:100]
Ytest = Ytest_[3:100]

# dIn: input delays so neural network can be used for systems where the output 
# depends on not only the current input but also previous inputs

net = prn.CreateNN([1,3,3,1])

net = prn.train_LM(P,Y,net,verbose=True,k_max=200,E_stop=1e-3)

# To investigate the influence of using the previous data P0test and Y0test, 
# we calculate the neural network output ytest0 with and ytest without using them
y = prn.NNOut(P,net)
y0test = prn.NNOut(Ptest,net)

fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
fs=18

# Train Data
ax0.set_title('Train Data',fontsize=fs)
ax0.plot(P,y,color='b',lw=2,label='NN Output')
ax0.plot(P,Y,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Train Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()

# Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(y0test,color='g',lw=2,label='NN Output with P0,Y0')
ax1.plot(Ytest,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.legend(fontsize=fs-2,loc='lower right')
ax1.grid()

fig.tight_layout()
plt.show()