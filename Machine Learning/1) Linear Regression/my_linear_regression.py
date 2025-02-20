#y = b0 + b1*x
#b0 = constant(bias) = y eksenini kestiği nokta
#b1 = coeff = eğim
#residual = y - y_head  / y= gerçek değer , y_head= tahmin edilen y
#-> residual +ve- çıkar bunun karesini alırsak - ler yok olur 
#MSE = sum(residual**2)/n   n= sample sayısı
#MSE = mean squarred error     (hata değeri)
#amaç = min(MSE)


# import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import data
df = pd.read_csv("linear_regression_dataset.csv",sep=";" )

#plot data
#plt.scatter(df.deneyim,df.maas) , plt.xlabel("deneyim"), plt.ylabel("maas"),plt.show() 

#%%linear regression

# sklearn library
from sklearn.linear_model import LinearRegression
# linear regression model
linear_reg = LinearRegression()

x= df.deneyim.values.reshape(-1,1)
y= df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%% prediction 

b0= linear_reg.predict([[0]])
print("b0: ", b0 )

#b0= linear_reg.intercept_  # y eksenini kestiği nokta
#print("b0: ", b0 )

b1 = linear_reg.coef_   #eğim 
print("b1: ", b1)

# maas = 1663 + 1138*deneyim

maas_yeni = 1663 + 1138*11
print(maas_yeni)

print(linear_reg.predict([[11]]))

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) # deneyim
plt.scatter(x,y,color="blue", label= "veri noktaları" )

y_head= linear_reg.predict(array) # maas
plt.plot(array,y_head,color= "red", label= "regression doğrusu")
plt.legend(), plt.show()
