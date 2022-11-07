
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X = data.data
y = data.target

import numpy as np
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.3)

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(x_train,y_train)
ypred = model.predict(x_test)

print("The accuracy score of KNN is: ", accuracy_score(ypred, y_test))

diff = ypred - y_test
missamplified = sum(abs(diff))
print("The total samples classified are: ",len(ypred))
print("The missamplified data are: ",missamplified)
print("Correct samplified data are: ",len(ypred)-missamplified)
