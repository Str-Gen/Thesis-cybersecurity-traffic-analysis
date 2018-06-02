#! /usr/bin/python2
import math
import itertools
import random
import operator
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import argparse
from collections import OrderedDict
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# %matplotlib inline

parser = argparse.ArgumentParser()
parser.add_argument("-F", "--features", action="store", dest="F",
                    help="Number of features to use", type=int, choices=[14, 16, 41], required=True)
parser.add_argument("-A", "--algorithm", action="store", dest="A", help="Which algorithm to use",
                    type=str, choices=["kNN", "DTree", "linSVC", "RForest", "binLR"], required=True)
results = parser.parse_args()

# 41, 16 or 14 Features
# 16 after one hot encoding leads to 95 features
# 41 after one hot encoding leads to 122 features
F = results.F

A = results.A

gt0 = time()

# Raw data
train20_nsl_kdd_dataset_path = "NSL_KDD_Dataset/KDDTrain+_20Percent.csv"
train_nsl_kdd_dataset_path = "NSL_KDD_Dataset/KDDTrain+.csv"
test_nsl_kdd_dataset_path = "NSL_KDD_Dataset/KDDTest+.csv"


# All columns
col_names = np.array(["duration", "protocol_type", "service", "flag", "src_bytes",
                      "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                      "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                      "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                      "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                      "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                      "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                      "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                      "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                      "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels", "labels_numeric"])

# All columns with nominal values (strings)
nominal_indexes = [1, 2, 3]
# All columns with binary values
binary_indexes = [6, 11, 13, 14, 20, 21]
# All other columns are numeric data, clever way of differencing, create range, transform it to a set and subtract the other indices, finally convert that to a list
numeric_indexes = list(set(range(41)).difference(
    nominal_indexes).difference(binary_indexes))

# Map the columns types to their name
# tolist is non-native python, it is available as a function on numpy ndarrays, which col_names is
nominal_cols = col_names[nominal_indexes].tolist()
binary_cols = col_names[binary_indexes].tolist()
numeric_cols = col_names[numeric_indexes].tolist()

dataframe = pd.read_csv(train_nsl_kdd_dataset_path, names=col_names)
dataframe = dataframe.drop('labels_numeric', axis=1)

if F == 14:
    relevant14 = np.array(['dst_bytes', 'wrong_fragment', 'count', 'serror_rate',
                           'srv_serror_rate', 'srv_rerror_rate', 'same_srv_rate', 'dst_host_count', 'dst_host_srv_count',
                           'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate'])
    relevant14 = np.append(relevant14, ['labels'])
    numeric_indexes = list(range(14))
    numeric_cols = relevant14[numeric_indexes].tolist()
    dataframe = dataframe[relevant14]

if F == 16:
    relevant16 = np.array(['service', 'flag', 'dst_bytes', 'wrong_fragment', 'count', 'serror_rate',
                           'srv_serror_rate', 'srv_rerror_rate', 'same_srv_rate', 'dst_host_count', 'dst_host_srv_count',
                           'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate'])
    relevant16 = np.append(relevant16, ['labels'])
    nominal_indexes = [0, 1]
    numeric_indexes = list(set(range(16)).difference(nominal_indexes))
    nominal_cols = relevant16[nominal_indexes].tolist()
    numeric_cols = relevant16[numeric_indexes].tolist()
    dataframe = dataframe[relevant16]
    #if A != 'DTree':
        # one hot encoding for categorical features
    for cat in nominal_cols:
        one_hot = pd.get_dummies(dataframe[cat])
        dataframe = dataframe.drop(cat, axis=1)
        dataframe = dataframe.join(one_hot)
if F == 41:
    #if A != 'DTree':
        # one hot encoding for categorical features
    for cat in nominal_cols:
        one_hot = pd.get_dummies(dataframe[cat])
        dataframe = dataframe.drop(cat, axis=1)
        dataframe = dataframe.join(one_hot)

# with pd.option_context('display.max_rows', 10, 'display.max_columns',None):
#     print dataframe

# Coarse grained dictionary of the attack types, every packet will be normal or is an attack, without further distinction
attack_dict_coarse = {
    'normal': 0,

    'back': 1,
    'land': 1,
    'neptune': 1,
    'pod': 1,
    'smurf': 1,
    'teardrop': 1,
    'mailbomb': 1,
    'apache2': 1,
    'processtable': 1,
    'udpstorm': 1,

    'ipsweep': 1,
    'nmap': 1,
    'portsweep': 1,
    'satan': 1,
    'mscan': 1,
    'saint': 1,

    'ftp_write': 1,
    'guess_passwd': 1,
    'imap': 1,
    'multihop': 1,
    'phf': 1,
    'spy': 1,
    'warezclient': 1,
    'warezmaster': 1,
    'sendmail': 1,
    'named': 1,
    'snmpgetattack': 1,
    'snmpguess': 1,
    'xlock': 1,
    'xsnoop': 1,
    'worm': 1,

    'buffer_overflow': 1,
    'loadmodule': 1,
    'perl': 1,
    'rootkit': 1,
    'httptunnel': 1,
    'ps': 1,
    'sqlattack': 1,
    'xterm': 1
}

dataframe["labels"] = dataframe["labels"].apply(
    lambda x: attack_dict_coarse[x])

# with pd.option_context('display.max_rows', 10, 'display.max_columns',None):
#     print dataframe["labels"]

# Standardization, current formula x-min / max-min
for c in numeric_cols:
    mean = dataframe[c].mean()
    stddev = dataframe[c].std()
    ma = dataframe[c].max()
    mi = dataframe[c].min()
    print c, "mean:", mean, "stddev:", stddev, "max:", ma, "mi:", mi
    dataframe[c] = dataframe[c].apply(lambda x: (x-mi)/(ma-mi))

with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
    print dataframe

label_loc = dataframe.columns.get_loc('labels')
array = dataframe.values
Y = array[:, label_loc]
X = np.delete(array, label_loc, 1)


def kNN_with_k_search(data, cross=0, k_start=1, k_end=101, k_step=2, distance_power=2):
    gt0 = time()
    for k in range(k_start, k_end, k_step):
        crossed[k] = []
    for k in range(k_start, k_end, k_step):
        sys.stdout.write('Round %d, k = %d \r' % (cross, k))
        sys.stdout.flush()
        classifier = KNeighborsClassifier(
            n_neighbors=k, p=distance_power, n_jobs=-1)
        classifier.fit(data['X_train'], data['Y_train'])
        result = classifier.score(data['X_test'], data['Y_test'])
        crossed[k].append([result, time()-gt0])
    return crossed


def kNN_with_k_fixed(data, k, distance_power):
    gt0 = time()
    crossed[k] = []
    classifier = KNeighborsClassifier(n_neighbors=k, p=distance_power)
    classifier.fit(data['X_train'], data['Y_train'])
    result = classifier.score(data['X_test'], data['Y_test'])
    crossed[k].append([result, time()-gt0])
    return crossed

def linSVC_with_tol_iter_search(data, cross=0, tol_start=0, tol_end=-9, iter_start=0, iter_end=7):    
    for tol_exp in range(tol_start,tol_end-1,-1):
        for iter_exp in range(iter_start, iter_end+1):
            gt0 = time()
            crossed['tol1e'+repr(tol_exp)+':iter1e'+repr(iter_exp)] = []
            classifier = LinearSVC(penalty='l2',tol=10**tol_exp, max_iter=10**iter_exp, dual=False)
            classifier.fit(data['X_train'],data['Y_train'])
            result = classifier.score(data['X_test'],data['Y_test'])
            crossed['tol1e'+repr(tol_exp)+':iter1e'+repr(iter_exp)].append([result,time()-gt0])
    return crossed

def linSVC_with_tol_iter_fixed(data,tolerance,iterations):
    gt0 = time()
    classifier = LinearSVC(penalty='l2', tol=tolerance,max_iter=iterations, dual=False)
    classifier.fit(X_train,Y_train)
    result = classifier.score(X_test,Y_test)
    crossed['linSVC'] = []
    crossed['linSVC'].append([result,time()-gt0])
    
def binLR_with_tol_iter_search(data,cross=0,tol_start=0,tol_end=-9,iter_start=0,iter_end=7):
    for tol_exp in range(tol_start,tol_end-1,-1):
        for iter_exp in range(iter_start, iter_end+1):
            sys.stdout.write('Testing tolerance = %d with iterations = %d \r' % (tol_exp, iter_exp))
            sys.stdout.flush()
            gt0 = time()
            crossed['tol1e'+repr(tol_exp)+':iter1e'+repr(iter_exp)] = []
            classifier = LogisticRegression(penalty='l2',tol=10**tol_exp, max_iter=10**iter_exp, dual=False,n_jobs=-1)
            classifier.fit(data['X_train'],data['Y_train'])
            result = classifier.score(data['X_test'],data['Y_test'])
            crossed['tol1e'+repr(tol_exp)+':iter1e'+repr(iter_exp)].append([result,time()-gt0])
    return crossed

def binLR_with_tol_iter_fixed(data,tolerance,iterations):
    gt0 = time()
    classifier = LogisticRegression(penalty='l2', tol=tolerance,max_iter=iterations, dual=False,n_jobs=-1)
    classifier.fit(X_train,Y_train)
    result = classifier.score(X_test,Y_test)
    crossed['binLR'] = []
    crossed['binLR'].append([result,time()-gt0])

def DTree_with_maxFeatures_maxDepth_search(data,cross=0,max_depth=5,max_features=251):    
    possible_features = range(1,max_features,1)
    possible_features.extend(['sqrt','log2',None])
    for md in range(1,max_depth,4):
        for mf in possible_features:
            sys.stdout.write('Testing max_depth = %d with max_features = %s \r' % (md, mf))
            sys.stdout.flush()
            gt0 = time()
            crossed['maxFeatures'+repr(mf)+':maxDepth'+repr(md)] = []
            classifier = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=md,max_features=mf)
            classifier.fit(data['X_train'],data['Y_train'])
            result = classifier.score(data['X_test'],data['Y_test'])
            crossed['maxFeatures'+repr(mf)+':maxDepth'+repr(md)].append([result,time()-gt0])
    return crossed

def DTree_with_maxFeatures_maxDepth_fixed(data,max_depth,max_features):
    gt0 = time()
    classifier = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=max_depth,max_features=max_features)
    classifier.fit(X_train,Y_train)
    result = classifier.score(X_test,Y_test)
    crossed['DTree:depth'+repr(max_depth)+':features'+repr(max_features)] = []
    crossed['DTree:depth'+repr(max_depth)+':features'+repr(max_features)].append([result,time()-gt0])

def RForest_with_maxFeatures_maxDepth_search(data,cross=0,max_depth=5,max_features=251):
    possible_features = range(1,max_features,1)
    possible_features.extend(['sqrt','log2',None])
    for md in range(1,max_depth,4):
        for mf in possible_features:
            sys.stdout.write('Testing max_depth = %d with max_features = %s \r' % (md, mf))
            sys.stdout.flush()
            gt0 = time()
            crossed['maxFeatures'+repr(mf)+':maxDepth'+repr(md)] = []
            classifier = RandomForestClassifier(criterion='gini',max_depth=md,max_features=mf,n_jobs=-1)
            classifier.fit(data['X_train'],data['Y_train'])
            result = classifier.score(data['X_test'],data['Y_test'])
            crossed['maxFeatures'+repr(mf)+':maxDepth'+repr(md)].append([result,time()-gt0])
    return crossed

def RForest_with_maxFeatures_maxDepth_fixed(data,max_depth,max_features):
    gt0 = time()
    classifier = RandomForestClassifier(criterion='gini',max_depth=max_depth,max_features=max_features,n_jobs=-1)
    classifier.fit(X_train,Y_train)
    result = classifier.score(X_test,Y_test)
    crossed['RForest:depth'+repr(max_depth)+':features'+repr(max_features)] = []
    crossed['RForest:depth'+repr(max_depth)+':features'+repr(max_features)].append([result,time()-gt0])

crossed = {}
for cross in range(0, 3):
    test_size = 0.33
    seed = int(round(random.random()*1000000))
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, test_size=test_size, random_state=seed)
    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test
    }    
    if A == 'kNN':
        crossed = kNN_with_k_search(data, cross=cross, k_start=1, k_end=5, k_step=2, distance_power=1)
    elif A == 'linSVC':
        crossed = linSVC_with_tol_iter_search(data,cross=cross, tol_start=0, tol_end=-9, iter_start=0, iter_end=7)
    elif A == 'binLR':
        crossed = binLR_with_tol_iter_search(data,cross=cross,tol_start=0, tol_end=-9, iter_start=0, iter_end=7)
    elif A == 'DTree':
        crossed = DTree_with_maxFeatures_maxDepth_search(data,cross=cross,max_depth=751,max_features=F)
    elif A == 'RForest':
        crossed = RForest_with_maxFeatures_maxDepth_search(data,cross=cross,max_depth=751,max_features=F)
print

for k in crossed:
    accs = [item[0] for item in crossed[k]]
    times = [item[1] for item in crossed[k]]
    crossed[k] = [np.mean(accs), np.std(accs), np.mean(times), np.std(times)]


validated = sorted(crossed.iteritems(), key=lambda (k,v): v[0],reverse=True)
for topn in range(0,len(crossed)) if len(crossed)<5 else range(0,5):
    # print '#',topn,'avg acc: {} stdev acc: {} avg time: {} stddev time: {}'.format(*validated[topn])
    print validated[topn]

''' 
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4)), 122 features
Top 4 results show that k=1 yields the highest accuracy 2h 37min 33s runtime intel core i5 4690 @3.5GHz
(1, [0.9923987299143654, 0.0, 112.55825209617615, 0.0])
(5, [0.9936255171750217, 0.0, 115.87367391586304, 0.0])
(9, [0.9934330799576638, 0.0, 117.78476095199585, 0.0])
(13, [0.9925911671317232, 0.0, 119.15998601913452, 0.0])
'''

'''
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4)), 95 features
Top 4 results show that k=1 yields the highest accuracy in 1h 53m 29s runtime intel core i5 4690 @3.5GHz
(1, [0.9892475704801309, 0.0, 79.29528713226318, 0.0])
(5, [0.9889348600019243, 0.0, 82.02173614501953, 0.0])
(9, [0.9886462041758877, 0.0, 83.00117683410645, 0.0])
(13, [0.9880207832194746, 0.0, 83.99710416793823, 0.0])
'''

'''
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4)), 14 features
Top 4 results show that k=1 yields the highest accuracy 47min 43s runtime intel core i5 4690 @3.5GHz
(1, [0.9552824016164726, 0.0, 39.16615080833435, 0.0])
(5, [0.9393822765322813, 0.0, 39.58459210395813, 0.0])
(9, [0.9554507841816607, 0.0, 40.0942759513855, 0.0])
(13, [0.9554748388338304, 0.0, 40.25523495674133, 0.0])
'''


# Fine-grained dictionary, packets are normal or attacks, and the attacks are divided into 4 subcategories

# Dictionary that contains mapping of various attacks to the four main categories
attack_dict = {
    'normal': 'normal',

    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',

    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',

    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}

# OLD STEP
# newrows,newcols = dataframe.shape
# # print newrows,newcols
# one_promille_rowcount = int(round(newrows/1000))

# # For all the numerical columns, shave off the rows with the one promille highest and lowest values
# for c in numeric_cols:
#     one_percent_largest = dataframe.nlargest(one_promille_rowcount,c)
#     one_percent_smallest = dataframe.nsmallest(one_promille_rowcount,c)
#     largest_row_indices, _ = one_percent_largest.axes
#     smallest_row_indices, _ = one_percent_smallest.axes
#     to_drop = set(largest_row_indices) | set(smallest_row_indices)
#     dataframe = dataframe.drop(to_drop,axis=0)

# print dataframe.shape
