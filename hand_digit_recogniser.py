from PIL import Image
import numpy as np
import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

#training
x_train = mnist.train_images()
y_train = mnist.train_labels()

x_test = mnist.test_images()
y_test = mnist.test_labels()

print(x_train)
print(x_test)
print(y_train)

print(x_train.ndim)
print(x_train.shape)
x_train = x_train.reshape((-1,28*28))
print(x_train.shape)

clf = MLPClassifier(solver='adam', activation='relu',hidden_layer_sizes=(64,64))
clf.fit(x_train,y_train)

prediction = clf.predict(x_test)
acc = confusion_matrix(y_test,prediction)
print(acc)

def accuracy(confusion_matrix):
    diagnol = cm.trace()
    elements = cm.sum()
    return diagnol/elements
print(accuracy(acc))