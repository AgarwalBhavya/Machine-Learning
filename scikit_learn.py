import matplotlib.pyplot as plt
x=[i for i in range(10)]
print(x)

y=[2*i for i in range(10)]
print(y)
plt.plot(x,y)

#for saving model
from sklearn.externals import joblib
filename = 'model.sav'
joblib.dump(clf,filename)

#open
clf = joblib.load(filename)
