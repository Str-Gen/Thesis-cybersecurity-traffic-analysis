#! /usr/bin/python
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, col, create_map, lit

from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark_knn.ml.classification import KNNClassifier
from pyspark.ml.classification import LinearSVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier

from time import time
import numpy as np
import pandas as pd
import random
import itertools
import operator
import argparse
import sys

# This is a simple test app. Use the following command to run assuming you're in the spark-knn folder:
# SPARK_PRINT_LAUNCH_COMMAND=true spark-submit --py-files /home/dhoogla/Documents/UGent/spark-knn/python/dist/pyspark_knn-0.1-py3.6.egg --driver-class-path /home/dhoogla/Documents/UGent/spark-knn/spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-84aecdb78cb7338fb2e49254f6fdddf508d7273f.jar --jars /home/dhoogla/Documents/UGent/spark-knn/spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-84aecdb78cb7338fb2e49254f6fdddf508d7273f.jar --driver-memory 12g --num-executors 4 light_spark_NSL_KDD.py

# local[*] master, * means as many worker threads as there are logical cores on your machine
sc = SparkContext(appName='lightweight_knn_nslkdd', master='local[*]')
sc.setLogLevel('ERROR')

sqlContext = SQLContext(sc)

parser = argparse.ArgumentParser()
parser.add_argument("-F","--features",action="store",dest="F",help="Number of features to use",type=int,choices=[14,16,41],required=True)
parser.add_argument("-A","--algorithm",action="store",dest="A",help="Which algorithm to use",type=str,choices=["kNN","DTree","linSVC","RForest","binLR"],required=True)
results = parser.parse_args()

# 41, 16 or 14 Features 
# 16 after one hot encoding leads to 95 features
# 41 after one hot encoding leads to 122 features
F = results.F
A = results.A

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
numeric_indexes = list(set(range(41)).difference(nominal_indexes).difference(binary_indexes))

# Map the columns types to their name
# tolist is non-native python, it is available as a function on numpy ndarrays, which col_names is
nominal_cols = col_names[nominal_indexes].tolist()
binary_cols = col_names[binary_indexes].tolist()
numeric_cols = col_names[numeric_indexes].tolist()

pandas_df = pd.read_csv(train_nsl_kdd_dataset_path,names=col_names)
pandas_df = pandas_df.drop('labels_numeric',axis=1)

if F == 14:
    relevant14 = np.array(['dst_bytes','wrong_fragment','count','serror_rate',
    'srv_serror_rate','srv_rerror_rate','same_srv_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate'])
    relevant14 = np.append(relevant14,['labels'])
    numeric_indexes = list(range(14))
    numeric_cols = relevant14[numeric_indexes].tolist() 
    pandas_df = pandas_df[relevant14]

if F == 16:
    relevant16 = np.array(['service','flag','dst_bytes','wrong_fragment','count','serror_rate',
    'srv_serror_rate','srv_rerror_rate','same_srv_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate'])
    relevant16 = np.append(relevant16,['labels'])
    nominal_indexes = [0,1]
    numeric_indexes = list(set(range(16)).difference(nominal_indexes))
    nominal_cols = relevant16[nominal_indexes].tolist()
    numeric_cols = relevant16[numeric_indexes].tolist()
    pandas_df = pandas_df[relevant16]

    # one hot encoding for categorical features
    for cat in nominal_cols:
        one_hot = pd.get_dummies(pandas_df[cat])    
        pandas_df = pandas_df.drop(cat,axis=1)
        pandas_df = pandas_df.join(one_hot)

if F == 41:
    # one hot encoding for categorical features
    for cat in nominal_cols:
        one_hot = pd.get_dummies(pandas_df[cat])    
        pandas_df = pandas_df.drop(cat,axis=1)
        pandas_df = pandas_df.join(one_hot)
# with pd.option_context('display.max_rows', 10, 'display.max_columns',None):
#     print pandas_df

# Coarse grained dictionary of the attack types, every packet will be normal or is an attack, without further distinction
attack_dict_coarse = {
    'normal': 0.0,

    'back': 1.0,
    'land': 1.0,
    'neptune': 1.0,
    'pod': 1.0,
    'smurf': 1.0,
    'teardrop': 1.0,
    'mailbomb': 1.0,
    'apache2': 1.0,
    'processtable': 1.0,
    'udpstorm': 1.0,

    'ipsweep': 1.0,
    'nmap': 1.0,
    'portsweep': 1.0,
    'satan': 1.0,
    'mscan': 1.0,
    'saint': 1.0,

    'ftp_write': 1.0,
    'guess_passwd': 1.0,
    'imap': 1.0,
    'multihop': 1.0,
    'phf': 1.0,
    'spy': 1.0,
    'warezclient': 1.0,
    'warezmaster': 1.0,
    'sendmail': 1.0,
    'named': 1.0,
    'snmpgetattack': 1.0,
    'snmpguess': 1.0,
    'xlock': 1.0,
    'xsnoop': 1.0,
    'worm': 1.0,

    'buffer_overflow': 1.0,
    'loadmodule': 1.0,
    'perl': 1.0,
    'rootkit': 1.0,
    'httptunnel': 1.0,
    'ps': 1.0,
    'sqlattack': 1.0,
    'xterm': 1.0
}

# Label all normal = 0, all attacks = 1
pandas_df["labels"] = pandas_df["labels"].apply(lambda x: attack_dict_coarse[x])

# Standardization, current formula x-min / max-min
for c in numeric_cols:
    # mean = dataframe[c].mean()
    # stddev = dataframe[c].std()
    ma = pandas_df[c].max()
    mi = pandas_df[c].min()    
    pandas_df[c] = pandas_df[c].apply(lambda x: (x-mi)/(ma-mi))


# Replace any NaN with 0
pandas_df.fillna(0,inplace=True)

spark_df = sqlContext.createDataFrame(pandas_df)
# print(type(spark_df))

# print(len(spark_df.columns))
all_features = [ feature for feature in spark_df.columns if feature != 'labels' ]
# print(len(all_features))
assembler = VectorAssembler( inputCols=all_features, outputCol='features')
spark_df = assembler.transform(spark_df)

def makeDense(v):
    return Vectors.dense(v.toArray())
makeDenseUDF = udf(makeDense,VectorUDT())

spark_df = spark_df.withColumn('features',makeDenseUDF(spark_df.features))
spark_df_vectorized = spark_df.select('features','labels')
spark_df_vectorized = spark_df_vectorized.withColumn('label',spark_df_vectorized.labels.cast(DoubleType())).drop('labels')
# spark_df_vectorized.show(truncate=False)


def kNN_with_k_search(data, cross=0, k_start=1, k_end=101, k_step=2):
    gt0 = time()
    for k in range(k_start, k_end, k_step):
        crossed['knn:k'+repr(k)] = []
    for k in range(k_start, k_end, k_step):
        sys.stdout.write('Round %d, k = %d \r' % (cross, k))
        sys.stdout.flush()
        classifier = KNNClassifier(k=k, featuresCol='features', labelCol='label', topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30 )  # bufferSize=-1.0,   bufferSizeSampleSize=[1, 2, 3] 
        model = classifier.fit(data['scaled_train_df'])
        predictions = model.transform(data['scaled_cv_df'])
        evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
        metric = evaluator.evaluate(predictions)
        crossed['knn:k'+repr(k)].append([metric, time()-gt0])
    return crossed


def kNN_with_k_fixed(data, k):
    gt0 = time()
    crossed['knn:k'+repr(k)] = []
    classifier = KNNClassifier(k=k, featuresCol='features', labelCol='label', topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30 )  # bufferSize=-1.0,   bufferSizeSampleSize=[1, 2, 3] 
    model = classifier.fit(data['scaled_train_df'])
    predictions = model.transform(data['scaled_cv_df'])
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    metric = evaluator.evaluate(predictions)
    crossed['knn:k'+repr(k)].append([metric, time()-gt0])
    return crossed

def linSVC_with_tol_iter_search(data, cross=0, tol_start=0, tol_end=-9, iter_start=0, iter_end=7):    
    for tol_exp in range(tol_start,tol_end-1,-1):
        for iter_exp in range(iter_start, iter_end+1):
            sys.stdout.write('Round %d Testing tol = %f with iterations= %f \r' % (cross,10**tol_exp, 10**iter_exp))
            sys.stdout.flush()
            gt0 = time()
            crossed['linSVC:tol1e'+repr(tol_exp)+':iter1e'+repr(iter_exp)] = []
            classifier = LinearSVC(maxIter=10**iter_exp, tol=10**tol_exp)
            model = classifier.fit(data['scaled_train_df'])
            predictions = model.transform(data['scaled_cv_df'])
            evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
            metric = evaluator.evaluate(predictions)
            crossed['linSVC:tol1e'+repr(tol_exp)+':iter1e'+repr(iter_exp)].append([metric,time()-gt0])
    return crossed

def linSVC_with_tol_iter_fixed(data,tolerance,iterations):
    gt0 = time()
    crossed['linSVC:tol1e'+repr(tolerance)+':iter1e'+repr(iterations)] = []
    classifier = LinearSVC(maxIter=iterations, tol=tolerance)
    model = classifier.fit(data['scaled_train_df'])
    predictions = model.transform(data['scaled_cv_df'])
    evaluator =  BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    metric = evaluator.evaluate(predictions)
    crossed['linSVC:tol1e'+repr(tolerance)+':iter1e'+repr(iterations)].append([metric,time()-gt0])

def binLR_with_tol_iter_search(data,cross=0, tol_start=0, tol_end=-9, iter_start=0, iter_end=7):    
    for tol_exp in range(tol_start,tol_end-1,-1):
        for iter_exp in range(iter_start, iter_end+1):
            sys.stdout.write('Round %d Testing tol = %f with iterations= %f \r' % (cross,10**tol_exp, 10**iter_exp))
            sys.stdout.flush()
            gt0 = time()
            crossed['binLR:tol1e'+repr(tol_exp)+':iter1e'+repr(iter_exp)] = []
            classifier = LogisticRegression(maxIter=10**iter_exp, tol=10**tol_exp)
            model = classifier.fit(data['scaled_train_df'])
            predictions = model.transform(data['scaled_cv_df'])
            evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
            metric = evaluator.evaluate(predictions)
            crossed['binLR:tol1e'+repr(tol_exp)+':iter1e'+repr(iter_exp)].append([metric,time()-gt0])
    return crossed

def binLR_with_tol_iter_fixed():
    gt0 = time()
    crossed['binLR:tol1e'+repr(tolerance)+':iter1e'+repr(iterations)] = []
    classifier = LogisticRegression(maxIter=iterations, tol=tolerance)
    model = classifier.fit(data['scaled_train_df'])
    predictions = model.transform(data['scaled_cv_df'])
    evaluator =  BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    metric = evaluator.evaluate(predictions)
    crossed['binLR:tol1e'+repr(tolerance)+':iter1e'+repr(iterations)].append([metric,time()-gt0])

def DTree_with_maxFeatures_maxDepth_search(data,cross=0,max_depth=5,max_features=251):    
    possible_features = range(1,max_features,1)
    possible_features.extend([round(math.sqrt(F)),round(math.log2(F)),F])
    for md in range(1,max_depth,4):
        for mf in possible_features:
            sys.stdout.write('Round %d Testing max_depth = %d with max_features = %s \r' % (cross, md, mf))
            sys.stdout.flush()
            gt0 = time()
            crossed['DTree:maxFeatures'+repr(mf)+':maxDepth'+repr(md)] = []
            classifier = DecisionTreeClassifier(maxDepth=md,maxBins=mf,impurity='gini',maxMemoryInMB=1024)
            model = classifier.fit(data['scaled_train_df'])
            predictions = model.transform(data['scaled_cv_df'])
            evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
            metric = evaluator.evaluate(predictions)
            crossed['DTree:maxFeatures'+repr(mf)+':maxDepth'+repr(md)].append([metric,time()-gt0])
    return crossed

def DTree_with_maxFeatures_maxDepth_fixed(data,max_depth,max_features):
    gt0 = time()    
    crossed['DTree:depth'+repr(max_depth)+':features'+repr(max_features)] = []
    classifier = DecisionTreeClassifier(maxDepth=md,maxBins=mf,impurity='gini',maxMemoryInMB=1024)
    model = classifier.fit(data['scaled_train_df'])
    predictions = model.transform(data['scaled_cv_df'])
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    metric = evaluator.evaluate(predictions)
    crossed['DTree:depth'+repr(max_depth)+':features'+repr(max_features)].append([metric,time()-gt0])
    return crossed

def RForest_with_maxFeatures_maxDepth_search(data,cross=0,max_depth=5,max_features=251):    
    possible_features = range(1,max_features,1)
    possible_features.extend([round(math.sqrt(F)),round(math.log2(F)),F])
    for md in range(1,max_depth,4):
        for mf in possible_features:
            sys.stdout.write('Round %d Testing max_depth = %d with max_features = %s \r' % (cross, md, mf))
            sys.stdout.flush()
            gt0 = time()
            crossed['RForest:maxFeatures'+repr(mf)+':maxDepth'+repr(md)] = []
            classifier = RandomForestClassifier(maxDepth=md,maxBins=mf,impurity='gini',maxMemoryInMB=1024)
            model = classifier.fit(data['scaled_train_df'])
            predictions = model.transform(data['scaled_cv_df'])
            evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
            metric = evaluator.evaluate(predictions)
            crossed['RForest:maxFeatures'+repr(mf)+':maxDepth'+repr(md)].append([metric,time()-gt0])
    return crossed

def RForest_with_maxFeatures_maxDepth_fixed(data,max_depth,max_features):
    gt0 = time()    
    crossed['RForest:depth'+repr(max_depth)+':features'+repr(max_features)] = []
    classifier = RandomForestClassifier(maxDepth=md,maxBins=mf,impurity='gini',maxMemoryInMB=1024)
    model = classifier.fit(data['scaled_train_df'])
    predictions = model.transform(data['scaled_cv_df'])
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    metric = evaluator.evaluate(predictions)
    crossed['RForest:depth'+repr(max_depth)+':features'+repr(max_features)].append([metric,time()-gt0])
    return crossed
            
crossed = {}
for cross in range(0,3):
    seed = int(round(random.random()*1000000))
    split = (spark_df_vectorized.randomSplit([0.67, 0.33], seed=seed))

    scaled_train_df = split[0].cache()
    # scaled_train_df.show(truncate=False)
    scaled_cv_df = split[1].cache()     

    data = {
        'scaled_train_df' : scaled_train_df,
        'scaled_cv_df' : scaled_cv_df
    }
    
    if A == 'kNN':
        kNN_with_k_search(data,cross=cross,k_start=1,k_end=3,k_step=2,)
    elif A == 'linSVC':
        crossed = linSVC_with_tol_iter_search(data,cross=cross, tol_start=0, tol_end=-9, iter_start=0, iter_end=7)
    elif A == 'binLR':
        crossed = binLR_with_tol_iter_search(data,cross=cross,tol_start=0, tol_end=-9, iter_start=0, iter_end=7)
    elif A == 'DTree':
        crossed = DTree_with_maxFeatures_maxDepth_search(data,cross=cross,max_depth=751,max_features=F)
    elif A == 'RForest':
        crossed = RForest_with_maxFeatures_maxDepth_search(data,cross=cross,max_depth=751,max_features=F)    
        
for k in crossed:
    accs = [item[0] for item in crossed[k]]
    times = [item[1] for item in crossed[k]]
    crossed[k] = [np.mean(accs), np.std(accs), np.mean(times),np.std(times)]

validated = sorted(crossed.items(), key=lambda s:s[1] ,reverse=True)
for topn in range(0,len(crossed)) if len(crossed)<5 else range(0,5):
    # print '#',topn,'avg acc: {} stdev acc: {} avg time: {} stddev time: {}'.format(*validated[topn])
    print(validated[topn])

'''
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4)), 122 features (F41)
Top 4 results show that k=1 yields the highest accuracy 44min 25s runtime intel core i5 4690 @3.5GHz
(1, [0.9987006730941386, 0.0, 17.568329334259033, 0.0])
(5, [0.997514684762422, 0.0, 11.270951271057129, 0.0])
(9, [0.9965676175259428, 0.0, 13.70993185043335, 0.0])
(13, [0.9956051353821973, 0.0, 15.49851942062378, 0.0])
'''

'''
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4)), 95 features (F16)
Top 4 results show that k=1 yields the highest accuracy 39min 48s runtime intel core i5 4690 @3.5GHz
(1, [0.9941603871380875, 0.0, 16.67420744895935, 0.0])
(5, [0.9924833333810361, 0.0, 9.670020818710327, 0.0])
(9, [0.9910571606236108, 0.0, 11.946027755737305, 0.0])
(13, [0.9901976184745408, 0.0, 13.429111957550049, 0.0])
'''

'''
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4)), 14 features (F14)
Top 4 results show that k=1 yields the highest accuracy 28min 48s runtime intel core i5 4690 @3.5GHz
(1, [0.9828101897880964, 0.0, 9.665040016174316, 0.0])
(5, [0.9782750353252064, 0.0, 6.7242395877838135, 0.0])
(9, [0.9762068651232392, 0.0, 8.55370044708252, 0.0])
(13, [0.9743859583347232, 0.0, 9.539605140686035, 0.0])
'''


# OLD STEP
# newrows,newcols = pandas_df.shape
# one_promille_rowcount = int(round(newrows/1000))
# # For all the numerical columns, shave off the rows with the one promille highest and lowest values
# for c in numeric_cols:
#     one_percent_largest = pandas_df.nlargest(one_promille_rowcount,c)
#     one_percent_smallest = pandas_df.nsmallest(one_promille_rowcount,c)
#     largest_row_indices, _ = one_percent_largest.axes    
#     smallest_row_indices, _ = one_percent_smallest.axes
#     to_drop = set(largest_row_indices) | set(smallest_row_indices)    
#     pandas_df = pandas_df.drop(to_drop,axis=0)