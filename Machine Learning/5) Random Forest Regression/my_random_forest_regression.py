import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random-forest-regression-dataset.csv",sep=";",header=None )

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% random forest regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x, y)
x_ = np.arange(min(x),max(x),0.01).reshape(-1, 1)
y_head= rf.predict(x_)
#%% görselleştirme
plt.scatter(x, y, color="red",label="gerçek")
plt.plot(x_,y_head,color= "blue",label="tahmin")
plt.legend()
plt.show()