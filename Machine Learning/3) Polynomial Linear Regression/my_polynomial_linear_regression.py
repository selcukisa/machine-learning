"""
polynomial linear regression = y = b0 + b1*x1 + b2*x2**2 + b3*x3**3 ..

"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial-regression.csv", sep=";" )

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)
"""
plt.scatter(x, y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()
"""
#%% linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%% predict
y_head =  lr.predict(x)

plt.scatter(x, y, color='blue', label='Veri Noktaları')
plt.plot(x,y_head,color= "red", label= "regression doğrusu")
plt.legend(), plt.show()

#%% polynomial regression

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=4)

x_poly = pr.fit_transform(x)
#%% fit
lr2 = LinearRegression()
lr2.fit(x_poly,y)

#%%
y_head2 = lr2.predict(x_poly)

plt.scatter(x, y,color = "red" , label="veri noktaları")
plt.plot(x,y_head2,color="blue",label="poly" )
plt.legend(), plt.show()


