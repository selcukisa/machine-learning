import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis = 1,inplace= True )

x_data = data.drop(["diagnosis"],axis=1)
y = data.diagnosis.values

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y= data.diagnosis.values

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train, y_train)
print("score: " ,rf.score(x_test, y_test))

y_pred = rf.predict(x_test)
y_true = y_test

#%% confusion matrix
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_true, y_pred)

#%% cm görselleştirme
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()