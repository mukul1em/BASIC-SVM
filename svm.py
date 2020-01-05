import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv(r'F:\mukul ml\svm\svm\Social_Network_Ads.csv')
x=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(xtrain,ytrain)
ypred=classifier.predict(xtest)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)

