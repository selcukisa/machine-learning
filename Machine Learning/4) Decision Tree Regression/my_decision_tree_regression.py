"""
dallanarak soru sorup yes veya no şeklinde ilerleyen sınıflandırma çeşidi
"""
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

df = pd.read_csv("decision-tree-regression-dataset.csv",sep=";", header=None )

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x, y)


tree_reg.predict([[5.5]])
x_ = np.arange(min(x),max(x),0.01 ).reshape(-1, 1)
y_head = tree_reg.predict(x_)
#%% görselleştirme
plt.scatter(x, y,color="red",label="gerçek değerler" )
plt.xlabel("tribün level")
plt.ylabel("ücret")
plt.plot(x_,y_head,color="blue",label="tahminler")
plt.legend()
plt.show()
   