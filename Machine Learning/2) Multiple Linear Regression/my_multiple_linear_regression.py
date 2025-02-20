"""
simple linear regression = y= b0 + b1*x
                    maas = b0+b1*deneyim

multiple linear regression = b0 + b1*x1 + b2*x2 ...
                    maas = b0 + b1*deneyim + b2*yas ...
maas = dependent varible (bağımlı değişken)
deneyim,yaş = independent varible (bağımsız değişken)

b0,b1,b2 = ? amaç=min(MSE)
"""

import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple-linear-regression-dataset.csv", sep=";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

#%%

mlr = LinearRegression()
mlr.fit(x,y)

print("b0: ", mlr.intercept_)
print("b1,b2: " ,mlr.coef_ )

#prediction
mlr.predict(np.array([[10,35],[5,35]]))