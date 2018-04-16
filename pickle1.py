#! /usr/bin/python2

# Saving an ML model in a pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
# we no longer explicitly invoke pickle, no need to import
# import pickle

url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url,names=names)
# this is a very clean way to display a dataframe in pandas, no hacks involved
# with pandas.option_context('display.max_rows', None, 'display.max_columns',3):
#     print dataframe


# this converts the pandas dataframe to a numpy array
array = dataframe.values
# this prints said numpy array, in this example a matrix of matrices with 768 rows & 9 columns in each row
# print array

# get every row in X, but for those rows only columns 0->8, 8th not included, all the previous columns are the observations
X = array[:,0:8]
# do the same for Y, but only store the 8th, the final column is the class 0 or 1
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=test_size,random_state=seed)

# We'll be training a logistic regression
model = LogisticRegression()
model.fit(X_train,Y_train)

# Saving a model is serializing it, in Python often called pickling, same as the preservatory method for the vegetable
filename = 'finalized_model.pkl'
# This was the straightforward code invoking pickle directly, replaced by joblib
# The pickle was ---> 1293 Bytes
# pickle.dump(model,open(filename,'wb'))

# the joblib way ---> 815 Bytes
joblib.dump(model,filename)

# ... Time passes ...


# Get the model back from disk and unpickle it
# loaded_model = pickle.load(open(filename,'rb'))
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test,Y_test)
print result
