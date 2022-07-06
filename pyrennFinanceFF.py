import matplotlib.pyplot as plt
import pyrenn as prn
import dataUtils as du

df = du.save_to_csv_from_yahoo('AMZN', 2020, 1, 1, 2020, 12, 30)

P = df['Open'].values
Y = df['Close'].values
Ptest_ = df['Open'].values
Ytest_ = df['Adj Close'].values

# Use the first hundred values as Test Data
Ptest = Ptest_[1:100]
Ytest = Ytest_[1:100]

# Create the neural network
net = prn.CreateNN([1,3,3,1])

# Train the neural network
net = prn.train_LM(P,Y,net,verbose=True,k_max=200,E_stop=1e-3)

# Using the trained neural network
y = prn.NNOut(P,net)
y0test = prn.NNOut(Ptest,net)

# Representation
fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
fs=18

# Train Data
ax0.set_title('Train Data',fontsize=fs)
ax0.plot(y,color='b',lw=2,label='NN Output')
ax0.plot(Y,color='r',marker='None',linestyle=':',
        lw=3,markersize=8,label='Train Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()

# Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(y0test,color='g',lw=2,label='NN Output')
ax1.plot(Ytest,color='r',marker='None',linestyle=':',
        lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.legend(fontsize=fs-2,loc='lower right')
ax1.grid()

fig.tight_layout()
plt.show()