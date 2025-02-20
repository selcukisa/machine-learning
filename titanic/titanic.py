#%% KÜTÜPHANELER 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
#%% VERİ YÜKLEME İŞLEMLERİ
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_PassengerId = test_df["PassengerId"]

#%% Varible Description
#['PassengerId'(int), 'Survived'(int), 'Pclass'(int), 'Name'(obj), 'Sex'(obj), 'Age'(float), 'SibSp'(int),'Parch'(int), 'Ticket'(obj), 'Fare'(float), 'Cabin'(obj), 'Embarked'(obj)]

# categorical varible: surivived , sex , pclass , embarked, cabin, name, ticket,sibsp and parch
# numerical varible: age and passengerld

def bar_plot(varible):
    var = train_df[varible]
    varValue = var.value_counts()
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(varible)
    plt.show()
    print("{}: \n {}".format(varible, varValue))

category = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]
for c in category:
    bar_plot(c)
    
category2 = ["Cabin","Name","Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))
    
def plot_hist(varible):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[varible],bins=100)
    plt.xlabel(varible)
    plt.ylabel("Frequency")
    plt.title("{} distribution whit hist".format(varible))
    plt.show()
    
numericVar = ["Fare","Age","PassengerId"]
for n in numericVar:
    plot_hist(n)
    
#%% basit data analizi
"""
veriler arasındaki ilişkiler
"""  
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending = False)
                                                                                        
train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Survived",ascending = False)

#%%outlier
def detect_outliers(df,features):
    outlier_indices =[]
    for c in features:
        q1 = np.percentile(df[c],25)
        q3 = np.percentile(df[c],75)
        IQR = q3 - q1
        outlier_step = IQR *1.5
        outlier_list_col = df[(df[c] < q1 - outlier_step) | (df[c] > q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multipe_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    return multipe_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]

train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis =0),reset_index(drop = True )


