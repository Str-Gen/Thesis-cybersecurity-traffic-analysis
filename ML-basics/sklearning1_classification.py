#! /usr/bin/python2

# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
# generate 2d classification dataset
# X contains 100 samples with 2 features (random real-valued) per sample
# y contains the class (1 or 0) for each of the 100 samples
X, y = make_blobs(n_samples=100,centers=2,n_features=2,random_state=1)

# fit final model
model = LogisticRegression()
model.fit(X,y)

# generate some new samples for which we don't keep the randomly generated classification
Xnew, _ = make_blobs(n_samples=3,centers=2,n_features=2,random_state=1)
# predict a class
ynew = model.predict(Xnew)
for i in range(len(Xnew)):
    print "X=%s, Predicted=%s" % (Xnew[i],ynew[i])

Xnew, _ = make_blobs(n_samples=3,centers=2,n_features=2,random_state=1)
# predict the odds of assignment to the available classes
ynew = model.predict_proba(Xnew)
for i in range(len(Xnew)):
    print "X=%s, Predicted=%s" % (Xnew[i],ynew[i])
