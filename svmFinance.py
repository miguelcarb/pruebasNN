from cProfile import label
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dataUtils as du


# @article{scikit-learn,
#  title={Scikit-learn: Machine Learning in {P}ython},
#  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
#          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
#          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
#          Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
#  journal={Journal of Machine Learning Research},
#  volume={12},
#  pages={2825--2830},
#  year={2011}
# }

df = du.get_df_from_csv('AMZN')

df.index = pd.to_datetime(df['Date'])

df = df.drop(['Date'], axis='columns')

df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low

# Store predictor variables
x = df[['Open-Close', 'High-Low']]
print(x.head())

# Target variables
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
print(y)

split = int(0.9*len(df))

# Train data set
x_train = x[:split]
y_train = y[:split]

#Test data set
x_test = x[split:]
y_test = y[split:]

model = SVC().fit(x_train, y_train)

df['sig'] = model.predict(x)
Z = df['sig'].values

print('Accuracy score: ', accuracy_score(y, Z))

df['Return'] = df.Close.pct_change()
df['Strategy_Return'] = df.Return * df.sig.shift(1)
df['Cum_Ret'] = df['Return'].cumsum()
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()

plt.plot(df['Cum_Ret'], color='red', label='Stock Return')
plt.plot(df['Cum_Strategy'], color='blue', label='Strategy Return')
plt.legend(fontsize=12,loc='lower right')

plt.show()