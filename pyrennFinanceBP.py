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

# define the first 3 timesteps t=[0,1,2] of Test Data as previous (known) data P0test and Y0test
P0test = Ptest_[0:3]
Y0test = Ytest_[0:3]

# Use the timesteps t = [3..99] as Test Data
Ptest = Ptest_[3:100]
Ytest = Ytest_[3:100]

# dIn: input delays so neural network can be used for systems where the output 
# depends on not only the current input but also previous inputs

"""
dOut:
This allows to add a recurrent connection of the outputs y of the neural network
to it’s first layer (which is similar to a recurrent connection of the output of
the network to it’s input). Thereby the neural network can be used for systems
where the output depends not only on the inputs, but also on prevoius outputs (states)
"""
net = prn.CreateNN([1,3,3,1],dIn=[1,2],dIntern=[],dOut=[1,2,3])
"""
P (numpy.array) - Training input data set P, 2d-array of shape (R,Q) with R 
rows (=number of inputs) and Q columns (=number of training samples)
Y (numpy.array) - Training output (target) data set Y, 2d-array of shape 
(SM,Q) with SM rows (=number of outputs) and Q columns (=number of training samples)
"""
net = prn.train_LM(P,Y,net,verbose=True,k_max=200,E_stop=1e-3)

# To investigate the influence of using the previous data P0test and Y0test, 
# we calculate the neural network output ytest0 with and ytest without using them
ytest = prn.NNOut(Ptest,net)
y0test = prn.NNOut(Ptest,net,P0=P0test,Y0=Y0test)

fig = plt.figure(figsize=(11,7))
ax1 = fig.add_subplot(111)
fs=18

# Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(ytest,color='b',lw=2,label='NN Output without P0,Y0')
ax1.plot(y0test,color='g',lw=2,label='NN Output with P0,Y0')
ax1.plot(Ytest,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.legend(fontsize=fs-2,loc='lower right')
ax1.grid()

fig.tight_layout()
plt.show()