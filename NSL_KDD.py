#! /usr/bin/python2
import math
import itertools
import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from collections import OrderedDict
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
    'normal': 'normal',
    
    'back': 'attack',
    'land': 'attack',
    'neptune': 'attack',
    'pod': 'attack',
    'smurf': 'attack',
    'teardrop': 'attack',
    'mailbomb': 'attack',
    'apache2': 'attack',
    'processtable': 'attack',
    'udpstorm': 'attack',
    
    'ipsweep': 'attack',
    'nmap': 'attack',
    'portsweep': 'attack',
    'satan': 'attack',
    'mscan': 'attack',
    'saint': 'attack',

    'ftp_write': 'attack',
    'guess_passwd': 'attack',
    'imap': 'attack',
    'multihop': 'attack',
    'phf': 'attack',
    'spy': 'attack',
    'warezclient': 'attack',
    'warezmaster': 'attack',
    'sendmail': 'attack',
    'named': 'attack',
    'snmpgetattack': 'attack',
    'snmpguess': 'attack',
    'xlock': 'attack',
    'xsnoop': 'attack',
    'worm': 'attack',
    
    'buffer_overflow': 'attack',
    'loadmodule': 'attack',
    'perl': 'attack',
    'rootkit': 'attack',
    'httptunnel': 'attack',
    'ps': 'attack',    
    'sqlattack': 'attack',
    'xterm': 'attack'
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



