#! /usr/bin/python
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, col, create_map, lit

from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark_knn.ml.classification import KNNClassifier

from time import time
import numpy as np
import pandas as pd
import random
import itertools
import operator

# This is a simple test app. Use the following command to run assuming you're in the spark-knn folder:
# spark-submit --py-files python/dist/pyspark_knn-0.1-py3.6.egg --driver-class-path spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-*.jar --jars spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-*.jar YOUR-SCRIPT.py

# local[*] master, * means as many worker threads as there are logical cores on your machine
sc = SparkContext(appName='lightweight_knn_nslkdd', master='local[*]')
# sc.setLogLevel('ERROR')

sqlContext = SQLContext(sc)

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

pandas_df = pd.read_csv(train_nsl_kdd_dataset_path,names=col_names)

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


newrows,newcols = pandas_df.shape
one_promille_rowcount = int(round(newrows/1000))
# For all the numerical columns, shave off the rows with the one promille highest and lowest values
for c in numeric_cols:
    one_percent_largest = pandas_df.nlargest(one_promille_rowcount,c)
    one_percent_smallest = pandas_df.nsmallest(one_promille_rowcount,c)
    largest_row_indices, _ = one_percent_largest.axes    
    smallest_row_indices, _ = one_percent_smallest.axes
    to_drop = set(largest_row_indices) | set(smallest_row_indices)    
    pandas_df = pandas_df.drop(to_drop,axis=0)

# Standardization, current formula x-min / max-min
for c in numeric_cols:
    # mean = dataframe[c].mean()
    # stddev = dataframe[c].std()
    ma = pandas_df[c].max()
    mi = pandas_df[c].min()    
    pandas_df[c] = pandas_df[c].apply(lambda x: (x-mi)/(ma-mi))


# one hot encoding for categorical features
for cat in nominal_cols:
    one_hot = pd.get_dummies(pandas_df[cat])    
    pandas_df = pandas_df.drop(cat,axis=1)
    pandas_df = pandas_df.join(one_hot)

# Replace any NaN with 0
pandas_df.fillna(0,inplace=True)

spark_df = sqlContext.createDataFrame(pandas_df)
# print(type(spark_df))
spark_df.drop('labels_numeric').collect()

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


crossed = {}
for cross in range(0,11):
    seed = int(round(random.random()*1000000))
    split = (spark_df_vectorized.randomSplit([0.8, 0.2], seed=seed))

    scaled_train_df = split[0].cache()
    # scaled_train_df.show(truncate=False)
    scaled_cv_df = split[1].cache()     
    
    for k in range(1,101,4):
        crossed[k] = []
        gt0 = time()
        print('Initializing')
        knn = KNNClassifier(k=k, featuresCol='features', labelCol='label', topTreeSize=1, topTreeLeafSize=1, subTreeLeafSize=1 )  # bufferSize=-1.0,   bufferSizeSampleSize=[1, 2, 3] 
        # print('Params:', [p.name for p in knn.params])
        print('Fitting:')
        model = knn.fit(scaled_train_df)
        # print('bufferSize:', model._java_obj.getBufferSize())
        # scaled_cv_df.show(truncate=False)
        # Don't drop label, need for verification!
        # scaled_cv_df = scaled_cv_df.drop('label')
        print('Predicting:')
        predictions = model.transform(scaled_cv_df)
        print('Predictions done:')
        # for row in predictions.collect():
        #     print(row)

        evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
        metric = evaluator.evaluate(predictions)

        print(metric)
        crossed[k].append([metric,time()-gt0])

for k in crossed:
    accs = [item[0] for item in crossed[k]]
    times = [item[1] for item in crossed[k]]

    crossed[k] = [np.mean(accs), np.std(accs), np.mean(times),np.std(times)]

validated = sorted(crossed.items(),key=operator.itemgetter(0))
for topn in range(4):
    print(validated[topn])

'''
3h 7m 24s runtime: 10 rounds of validation, testing k=1->k=97
After 10 fold cross validation it turns out that only looking at the closest neighbour (k=1) yields the best result
(1, [0.998621590656475, 0.0, 24.55202031135559, 0.0])
(5, [0.9976535562473619, 0.0, 17.84294605255127, 0.0])
(9, [0.9969919818270346, 0.0, 20.179744720458984, 0.0])
(13, [0.9965387989030701, 0.0, 21.47579026222229, 0.0])
'''