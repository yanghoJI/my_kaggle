from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot
from numpy import random
import time

random.seed(2018)

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

#print(dataset[:5])
#split data
X = dataset[:, 0:8]
y = dataset[:, 8]

mask = random.rand(len(X)) < 0.8
X_train = X[mask]
Y_train = y[mask]
X_val = X[~mask]
Y_val = y[~mask]

t1 = time.time()
model = XGBClassifier(n_estimators=100, booster='gbtree')
print(model)
model.fit(X_train, Y_train)
result = model.predict(X_val)
print('validation acc :{}\t runtime : {:.3f}'.format(sum(Y_val == result) / len(Y_val), time.time()-t1))
print('pridiction : {}'.format(result[:10]))
print('pridiction : {}'.format(Y_val[:10]))
#print(result.shape)
##plot_importance(model)
#pyplot.show()