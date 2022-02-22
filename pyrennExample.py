import pandas as pd
import matplotlib.pyplot as plt
import pyrenn as prn

df = pd.ExcelFile('/Users/PCComp/Documents/TFG/example_data.xlsx').parse('friction')
P = df.loc[0:40]['P']
Y = df.loc[0:40]['Y']
Ptest = df['Ptest'].values
Ytest = df['Ytest'].values

# feedforward network with 1 innput and 1 output and 2 hidden layers

net = prn.CreateNN([1,3,3,1])

net = prn.train_LM(P,Y,net,verbose=True,k_max=100,E_stop=1e-5)

y = prn.NNOut(P,net)
ytest = prn.NNOut(Ptest,net)

fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
fs=18

#Train Data
ax0.set_title('Train Data',fontsize=fs)
ax0.plot(P,y,color='b',lw=2,label='NN Output')
ax0.plot(P,Y,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Train Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()

#Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(Ptest,ytest,color='b',lw=2,label='NN Output')
ax1.plot(Ptest,Ytest,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.legend(fontsize=fs-2,loc='upper left')
ax1.grid()

fig.tight_layout()
plt.show()