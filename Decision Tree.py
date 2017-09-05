import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn import tree

balance_data=pd.read_csv('/Users/mananmanwani/Python Projects/DecisionTree.data.txt')
print ("Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)
balance_data.head()

X=balance_data.values[:,1:5]
y=balance_data.values[:,0]
y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)

clf=DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=3, min_samples_leaf=5)
clf.fit(X_train, y_train)

predict=clf.predict(X_test)

accuracy=accuracy_score(y_test,predict)*100
print (accuracy)