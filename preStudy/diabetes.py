from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

#print(dataset[:5])

X = dataset[:, 0:8]
y = dataset[:, 8]

model = XGBClassifier(n_estimators=100, booster='gbtree')
print(model)
model.fit(X, y)
result = model.predict(X)
print('training acc :{}'.format(sum(y == result) / len(y)))
print('pridiction : {}'.format(result[:10]))
print('pridiction : {}'.format(y[:10]))
print(result.shape)
##plot_importance(model)
#pyplot.show()