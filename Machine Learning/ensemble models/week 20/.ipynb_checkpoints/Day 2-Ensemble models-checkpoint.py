# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

x  = np.random.randint(10,20,(3,5))
print(x)

df = pd.read_csv("../../../datasets/titanic/train.csv")

from sklearn.datasets import load_breast_cancer

X,y=load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y)


bagging_model1=BaggingClassifier(base_estimator=KNeighborsClassifier())
                  
bagging_model1.fit(X_train,y_train)
                  
print(bagging_model1.score(X_train,y_train))
print(bagging_model1.score(X_test,y_test))
print()
print()


bagging_model2=BaggingClassifier(base_estimator=DecisionTreeClassifier())
                  
bagging_model2.fit(X_train,y_train)
                  
print(bagging_model2.score(X_train,y_train))
print(bagging_model2.score(X_test,y_test))

print()
print()


bagging_model3=BaggingClassifier(base_estimator=LogisticRegression(solver="liblinear"))
                  
bagging_model3.fit(X_train,y_train)
                  
print(bagging_model3.score(X_train,y_train))
print(bagging_model3.score(X_test,y_test))






