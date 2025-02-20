# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:14:08 2024

@author: Administrator
"""

#%% kütüphanelar
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#%% veri işleme
data = pd.read_csv("mushroom_cleaned.csv")
#print(data.info())
y= data.clas.values
x_data = data.drop(["clas"],axis=1)


# %% normalization
x = (x_data - np.min(x_data))/ (np.max(x_data)-np.min(x_data))
# %% train test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

x_train = x_train.T
x_test =  x_test.T
y_train = y_train.T
y_test =  y_test.T


# %% parameter initialize and sigmoid function

def weights_and_bias(boyut):
    w = np.full((boyut,1),0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head

# %%forward_backward_propagation
def forward_backward_propagation(w,b,x_train,y_train):
    #forward
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -(1-y_train)*np.log(1-y_head)-y_train*np.log(y_head)
    cost = np.sum(loss)/x_train.shape[1]
    
    #backward
    turev_weight = (np.dot(x_train,((y_head-y_train).T)))/ x_train.shape[1]
    turev_bias =   np.sum(y_head-y_train)/x_train.shape[1]
    turev = {"turev_weight":turev_weight ,"turev_bias":turev_bias }
    return cost,turev

#%% Updating(learning) parameters

def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    cost_list = []  #her hata
    cost2_list = [] # 10hata ortalaması
    index =[]  #kaç defa
    
    for i in range(number_of_iteration):
        cost,turev = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        w = w - learning_rate * turev["turev_weight"]
        b = b - learning_rate * turev["turev_bias"]
        if i % 10 == 0 :
            cost2_list.append(cost)
            index.append(i)
            print("yinelemeden sonraki maliyet %i : %f " %(i,cost))
            
    parameters = {"weights": w , "bias": b}
    plt.plot(index, cost2_list)
    plt.xticks(index,rotation= "vertical" )
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters,turev, cost_list
        
    
#%%  # prediction
def tahmin(w,b,x_test):
    z = (np.dot(w.T,x_test)+b)
    y_tahmin = np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_tahmin[0,i] = 0
            
        else:
            y_tahmin[0,i] = 1
    return y_tahmin

# %% logistic_regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    boyut =  x_train.shape[0]  # that is 30
    w,b = weights_and_bias(boyut)
    # do not change learning rate
    parameters, turev, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = tahmin(parameters["weights"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300) 


#%%kütüphane kullanarak
#%% sklearn with LR
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))



