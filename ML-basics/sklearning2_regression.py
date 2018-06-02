#! /usr/bin/python2

# example of training a linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# generate a random dataset
X, y, coef = make_regression(n_samples=100, n_features=2, noise=0.1, coef=True)
print 'X '+repr(X)
print 'y '+repr(y)
print "z = %s x + %s" % (coef[0],coef[1])

# fit the model
model = LinearRegression()
model.fit(X,y)

# generate data to make predictions
Xnew, _, cf = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1, coef=True)
print 'X '+repr(X)
print "z = %s x + %s" % (cf[0],cf[1])

# make a prediction
ynew = model.predict(Xnew)

# show the predictions
for i in range(len(Xnew)):
    print "X=%s, Predicted=%s" % (Xnew[i],ynew[i])

