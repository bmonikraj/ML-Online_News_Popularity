import os
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
import numpy

dirPath = os.getcwd()
dataSet = pandas.read_csv(str(dirPath)+"\\OnlineNewsPopularity.csv")
#Read CSV dataset

Ydata = dataSet[' shares']
del dataSet[' shares']
Xdata = dataSet
del Xdata['url']
del Xdata[' timedelta']
X = numpy.array(Xdata)
Y = numpy.array(Ydata)
#Removed non predictive data features and target feature manually from data set

print X.shape
print Y.shape

rf = RandomForestRegressor()
rf.fit(X,Y)
print rf.feature_importances_
#To print the best features available for regression, in feature selection.. We will consider all features

treeRegressor = DecisionTreeRegressor()
treeRegressor = treeRegressor.fit(X,Y)

Y_res = treeRegressor.predict(X)
ErrorY = numpy.array(numpy.subtract(Y,Y_res))

error = 0
for k in range(0,ErrorY.size,1):
    if ErrorY[k]!=0:
        error=error+1
sizeOfErrorY = ErrorY.size
error = float(float(error)/float(sizeOfErrorY))
error = error*float(100)
print "Error on training set Decision Regressor is "+str(error)

linearReg = Ridge(alpha=0.5)
linearReg.fit(X,Y)

Y_res = linearReg.predict(X)
ErrorY = numpy.array(numpy.subtract(Y,Y_res))

error = 0
for k in range(0,ErrorY.size,1):
    if ErrorY[k]!=0:
        error=error+1
sizeOfErrorY = ErrorY.size
error = float(float(error)/float(sizeOfErrorY))
error = error*float(100)
print "Error on training set Linear Regression is "+str(error)
print Y
print Y_res


