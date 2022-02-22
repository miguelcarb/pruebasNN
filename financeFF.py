import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import numpy as np
import neurolab as nl
import pandas as pd
import pandas_datareader.data as web

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

AMZN = get_df_from_csv('AMZN')

# tarea: dividir inputs en arrays de 4 con el Adj Close semanal*
######## dividir a su vez esos arrays de 4 en arrays de 3 para datos de entrenamiento (Xi) y 1 para datos de testeo (Yi)
######## 252 ejemplos de entrenamiento (mod 4 = 0)
######## hidden layer de tres neuronas, input layer de tres neuronas, output layer de una neurona
######## tasa de aprendizaje = 0.01

# de momento semanal, con 6 y 1

# data

sample = np.array(df['Adj Close'])
size = len(sample)
sampleDiv = sample.reshape(36, 7) # dividir en arrays de 7 (semanalmente)

# train examples

# get six first values of each array in sampleDiv
sampleDiv_train = sampleDiv[0:36, 0:6]

# get last value of each array in sampleDiv
sampleDiv_test = sampleDiv[0:36, 6:7]

minInput = np.min(sampleDiv_train)
maxInput = np.max(sampleDiv_train)
minOutput = np.min(sampleDiv_test)
maxOutput = np.max(sampleDiv_test)

plt.subplot(211)
plt.plot(sampleDiv_test, color='#5a7d9a', marker='o', label='target')

net = nl.net.newff([[minInput, maxInput], [minInput, maxInput], [minInput, maxInput], [minInput, maxInput], [minInput, maxInput], [minInput, maxInput]], [6, 6, 1])

print(len(sampleDiv_train))
print(len(sampleDiv_test))

error = net.train(sampleDiv_train, sampleDiv_test, epochs=500, show=100, goal=0.01)

#   simular la red
out = net.sim(sampleDiv_train)
plt.plot(out, color='#adad3b', marker='p', label='output')
plt.legend()

plt.subplot(212)
plt.plot(error)
plt.xlabel('numero de epocas')
plt.ylabel('error de entrenamiento')
plt.grid()

plt.show()

"""
# [[1, 2, 3], [1, 2, 3]] -> [[1, 1], [2, 2], [3, 3]]

input = np.column_stack((df['Open'], df['Close']))

"""