from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

#split it in features and labels
x = iris.data
y = iris.target

classes = ['Iris setosa','Iris versicolor','Iris verginica']

print(x.shape)
print(y.shape)

#hours of study vs good/bad grades
#10 different students
#train with 8 students
#predict with remaining 2
#level of accuracy

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2)

model = svm.SVC()
model.fit(x_train,y_train)

print(model)
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test,predictions)

print(predictions)
print(accuracy)