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
from collections import OrderedDict
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier

# %matplotlib inline
gt0 = time()

# Raw data
train20_nsl_kdd_dataset_path = "NSL_KDD_Dataset/KDDTrain+_20Percent.csv"
train_nsl_kdd_dataset_path = "NSL_KDD_Dataset/KDDTrain+.csv"
test_nsl_kdd_dataset_path = "NSL_KDD_Dataset/KDDTest+.csv"

# All columns
col_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","labels_numeric"])



# All columns with nominal values (strings)
nominal_indexes = [1,2,3]
# All columns with binary values
binary_indexes = [6,11,13,14,20,21]
# All other columns are numeric data, clever way of differencing, create range, transform it to a set and subtract the other indices, finally convert that to a list
numeric_indexes = list(set(range(41)).difference(nominal_indexes).difference(binary_indexes))

# Map the columns types to their name
# tolist is non-native python, it is available as a function on numpy ndarrays, which col_names is
nominal_cols = col_names[nominal_indexes].tolist()
binary_cols = col_names[binary_indexes].tolist()
numeric_cols = col_names[numeric_indexes].tolist()

dataframe = pandas.read_csv(train20_nsl_kdd_dataset_path,names=col_names)

with pandas.option_context('display.max_rows', 10, 'display.max_columns',None):
    print dataframe

# Coarse grained dictionary of the attack types, every packet will be normal or is an attack, without further distinction
attack_dict_coarse = {
    'normal': 0,
    
    'back': 1,
    'land': 1,
    'neptune': 1,
    'pod': 1,
    'smurf': 1,
    'teardrop': 1,
    'mailbomb':1,
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

dataframe["labels"] = dataframe["labels"].apply(lambda x: attack_dict_coarse[x])

with pandas.option_context('display.max_rows', 10, 'display.max_columns',None):
    print dataframe["labels"]

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


# one hot encoding for categorical features
for cat in nominal_cols:
    one_hot = pd.get_dummies(dataframe[cat])    
    dataframe = dataframe.drop(cat,axis=1)
    dataframe = dataframe.join(one_hot)

newrows,newcols = dataframe.shape
print newrows,newcols
one_promille_rowcount = int(round(newrows/1000))

# For all the numerical columns, shave off the rows with the one promille highest and lowest values
for c in numeric_cols:
    one_percent_largest = dataframe.nlargest(one_promille_rowcount,c)
    one_percent_smallest = dataframe.nsmallest(one_promille_rowcount,c)
    largest_row_indices, _ = one_percent_largest.axes    
    smallest_row_indices, _ = one_percent_smallest.axes
    to_drop = set(largest_row_indices) | set(smallest_row_indices)    
    dataframe = dataframe.drop(to_drop,axis=0)
    

print dataframe.shape

# Standardization, current formula x-min / max-min
for c in numeric_cols:
    mean = dataframe[c].mean()
    stddev = dataframe[c].std()
    ma = dataframe[c].max()
    mi = dataframe[c].min()    
    print c,"mean:",mean,"stddev:",stddev,"max:",ma,"mi:",mi
    dataframe[c] = dataframe[c].apply(lambda x: (x-mi)/(ma-mi))

with pandas.option_context('display.max_rows', 10, 'display.max_columns',None):
    print dataframe[["src_bytes","dst_bytes"]]

label_loc = dataframe.columns.get_loc("labels")
print "label:",label_loc

array = dataframe.values
Y = array[:,label_loc]
X = np.delete(array,label_loc,1)

crossed = {}
for cross in range(0,11):
    test_size = 0.2
    seed = int(round(random.random()*1000000))
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=test_size,random_state=seed)

    for k in range(1,201,4):
        crossed[k] = []

    for k in range(1,201,4):
        sys.stdout.write('Round %d, k = %d \r' % (cross,k))        
        sys.stdout.flush()
        gt0 = time()
        neigh = KNeighborsClassifier(n_neighbors=k, p=1, n_jobs=-1)
        neigh.fit(X_train,Y_train)
        result = neigh.score(X_test,Y_test)
        crossed[k].append([result,time()-gt0])
print

for k in crossed:   
    accs = [item[0] for item in crossed[k]]
    times = [item[1] for item in crossed[k]]

    crossed[k] = [np.mean(accs), np.std(accs), np.mean(times), np.std(times)]
    

validated = sorted(crossed.items(),key=operator.itemgetter(0))
for topn in range(4):
    #print '#',topn,'avg acc: {} stdev acc: {} avg time: {} stddev time: {}'.format(*validated[topn])
    print validated[topn]

''' 
28min50s runtime for the entire run: 10 rounds of validation, testing k=1 -> k=197
After 10 fold cross validation it turns out that only looking at the closest neighbour (k=1) yields the best result
(1, [0.9980980557903635, 0.0, 1.7285821437835693, 0.0])
(5, [0.9959847844463229, 0.0, 2.0070080757141113, 0.0])
(9, [0.9955621301775148, 0.0, 2.2154479026794434, 0.0])
(13, [0.9945054945054945, 0.0, 2.309248208999634, 0.0])
'''
