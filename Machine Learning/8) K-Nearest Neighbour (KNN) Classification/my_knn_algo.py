#KNN = K en yakın komşu
"""
1- K değerini seç
2-K en yakın data noktalarını bul
3-K en yakın komşu arasında hangi classtan kaç tane var
4-test ettiğimiz point ya da data hangi classa ait tespit et

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis= 1,inplace = True )

M = data[data.diagnosis =="M"]
B = data[data.diagnosis =="B"]

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kötü huylu")
plt.scatter(B.radius_mean,B.texture_mean,color="blue",label="iyi huylu")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
x_data = data.drop(["diagnosis"],axis=1)
y = data.diagnosis.values

#%% normalization
x= (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%% train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

#%% knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11) #n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{} nn score: {}".format(3, knn.score(x_test,y_test)))
#%% find k value (en iyi değeri bulma)
knn_score= []
for each in range(1,15):    
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train, y_train)
    knn_score.append(knn2.score(x_test, y_test))
    
plt.plot(range(1,15),knn_score)
plt.xlabel("k_values")
plt.ylabel("accuracy")
plt.show()
    

