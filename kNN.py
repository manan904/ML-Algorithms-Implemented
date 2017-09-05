import numpy
from sklearn import preprocessing,cross_validation, neighbors
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('/Users/mananmanwani/Python Projects/iris.data.txt')
df.replace('?',-99999, inplace=True)
Color ={
    'Iris-virginica':'R',
    'Iris-setosa':'G',
    'Iris-versicolor':'B'
}
labels=['Iris-virginica','Iris-setosa','Iris-versicolor']
#df.drop(['id'],1,inplace=True)

plt.scatter(df['sepal_length'],df['sepal_width'],c=df['class'].apply(lambda x: Color[x]))
ax.margins(0.05)
plt.xlabel('Petal Width')
plt.ylabel('Petal Length')
plt.title('Petal Width vs Length')
plt.legend(labels)
plt.show()
X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)
print(accuracy)

example_measures=np.array([5.2,3.6,1.5,0.2])
example_measures=example_measures.reshape(1,-1)

prediction=clf.predict(example_measures)
print (prediction)


