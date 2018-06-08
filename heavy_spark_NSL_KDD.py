#! /usr/bin/python
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, min, max, lit, lower
import pyspark.sql.functions as sql
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoderEstimator, MinMaxScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark_knn.ml.classification import KNNClassifier
from pyspark.ml.classification import LinearSVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier 
from time import time
from datetime import timedelta
import numpy as np
import pandas as pd
import random
import itertools
import operator
import argparse
import math

totaltime = time()

# This is a simple test app. Use the following command to run assuming you're in the spark-knn folder:
# spark-submit --py-files python/dist/pyspark_knn-0.1-py3.6.egg --driver-class-path spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-*.jar --jars spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-*.jar YOUR-SCRIPT.py

# local[*] master, * means as many worker threads as there are logical cores on your machine
spark = SparkSession.builder \
        .master('local[*]') \
        .appName('spark_knn_nslkdd') \
        .getOrCreate()
        # .config('spark.driver.memory','12g') \
        # .config('spark.driver.maxResultSize','10g') \
        # .config('spark.executor.memory','4g') \        
        # .config('spark.executor.instances','4') \
        # .config('spark.executor.cores','4') \
        # .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

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
numeric_indexes = list(set(range(41)).difference(
    nominal_indexes).difference(binary_indexes))

# Map the columns types to their name
# tolist is non-native python, it is available as a function on numpy ndarrays, which col_names is
nominal_cols = col_names[nominal_indexes].tolist()
binary_cols = col_names[binary_indexes].tolist()
numeric_cols = col_names[numeric_indexes].tolist()

def read_dataset_typed(path):
    schema = StructType([
        StructField('duration', DoubleType(), True),
        StructField('protocol_type', StringType(), True),
        StructField('service', StringType(), True),
        StructField('flag', StringType(), True),
        StructField('src_bytes', DoubleType(), True),
        StructField('dst_bytes', DoubleType(), True),
        StructField('land', DoubleType(), True),
        StructField('wrong_fragment', DoubleType(), True),
        StructField('urgent', DoubleType(), True),
        StructField('hot', DoubleType(), True),
        StructField('num_failed_logins', DoubleType(), True),
        StructField('logged_in', DoubleType(), True),
        StructField('num_compromised', DoubleType(), True),
        StructField('root_shell', DoubleType(), True),
        StructField('su_attempted', DoubleType(), True),
        StructField('num_root', DoubleType(), True),
        StructField('num_file_creations', DoubleType(), True),
        StructField('num_shells', DoubleType(), True),
        StructField('num_access_files', DoubleType(), True),
        StructField('num_outbound_cmds', DoubleType(), True),
        StructField('is_host_login', DoubleType(), True),
        StructField('is_guest_login', DoubleType(), True),
        StructField('count', DoubleType(), True),
        StructField('srv_count', DoubleType(), True),
        StructField('serror_rate', DoubleType(), True),
        StructField('srv_serror_rate', DoubleType(), True),
        StructField('rerror_rate', DoubleType(), True),
        StructField('srv_rerror_rate', DoubleType(), True),
        StructField('same_srv_rate', DoubleType(), True),
        StructField('diff_srv_rate', DoubleType(), True),
        StructField('srv_diff_host_rate', DoubleType(), True),
        StructField('dst_host_count', DoubleType(), True),
        StructField('dst_host_srv_count', DoubleType(), True),
        StructField('dst_host_same_srv_rate', DoubleType(), True),
        StructField('dst_host_diff_srv_rate', DoubleType(), True),
        StructField('dst_host_same_src_port_rate', DoubleType(), True),
        StructField('dst_host_srv_diff_host_rate', DoubleType(), True),
        StructField('dst_host_serror_rate', DoubleType(), True),
        StructField('dst_host_srv_serror_rate', DoubleType(), True),
        StructField('dst_host_rerror_rate', DoubleType(), True),
        StructField('dst_host_srv_rerror_rate', DoubleType(), True),
        StructField('labels', StringType(), True),
        StructField('labels_numeric', DoubleType(), True)
        ])

    return spark.read.csv(path,schema=schema,mode='FAILFAST')

# A custom transform
class Attack2DTransformer(Transformer):    
    def __init__(self):
        super(Attack2DTransformer,self).__init__()
    def _transform(self,dataset):
        # regex: match full line ^$, match any character any nr of times .*, look ahead and if the word normal is matched, then fail the match
        # what it does: everything on a line in the labels column that isn't the word normal replace with the word attack, removing the distinct categories of attacks
        return dataset.withColumn('2DAttackLabel',sql.regexp_replace(col('labels'),'^(?!normal).*$','attack'))

# Followed by a StringIndexer, which takes the available strings in a column ranks them by number of appearances and then gives each string a number
# 1.0 for the string with the most occurrences (default), 2.0 for 2nd most and so on
label2DIndexer = StringIndexer(inputCol='2DAttackLabel',outputCol='index_2DAttackLabel')

# Pack the 2D string transform and subsequent indexing transform into a unit, Pipeline
mapping2DPipeline = Pipeline(stages=[Attack2DTransformer(),label2DIndexer])

t0 = time()
train_df = read_dataset_typed(train_nsl_kdd_dataset_path)
train_df = mapping2DPipeline.fit(train_df).transform(train_df)
train_df = train_df.drop('labels','labels_numeric','2DAttackLabel')
train_df = train_df.withColumnRenamed('index_2DAttackLabel','label')
# train_df = train_df.cache()
# train_df.show(n=5,truncate=False,vertical=True)
#print(time()-t0)

t0 = time()
test_df = read_dataset_typed(test_nsl_kdd_dataset_path)
test_df = mapping2DPipeline.fit(test_df).transform(test_df)
test_df = test_df.drop('labels','labels_numeric','2DAttackLabel')
test_df = test_df.withColumnRenamed('index_2DAttackLabel','label')
test_df = test_df.cache()
# test_df.show(n=5,truncate=False,vertical=True)
#print(time()-t0)

if F == 14:
    t0 = time()
    relevant14 = np.array(['dst_bytes','wrong_fragment','count','serror_rate',
    'srv_serror_rate','srv_rerror_rate','same_srv_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate'])
    #relevant14 = np.append(relevant14,['label'])
    train_df = train_df.select(*relevant14,'label')
    numeric_cols = relevant14.tolist()
    nominal_cols = []
    #print(time()-t0)

if F == 16:
    t0 = time()
    relevant16 = np.array(['service','flag','dst_bytes','wrong_fragment','count','serror_rate',
    'srv_serror_rate','srv_rerror_rate','same_srv_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate'])
    #relevant16 = np.append(relevant16,['label'])
    train_df = train_df.select(*relevant16,'label')
    nominal_indexes = [0,1]
    numeric_indexes = list(set(range(16)).difference(nominal_indexes))
    nominal_cols = relevant16[nominal_indexes].tolist()
    numeric_cols = relevant16[numeric_indexes].tolist()
    idxs = [StringIndexer(inputCol=c,outputCol=c+'_index') for c in nominal_cols]
    ohes = [OneHotEncoderEstimator(inputCols=[c+'_index'],outputCols=[c+'_numeric'],dropLast=False) for c in nominal_cols]
    idxs.extend(ohes)
    OhePipeline = Pipeline(stages=idxs)
    train_df = OhePipeline.fit(train_df).transform(train_df)
    train_df = train_df.drop(*nominal_cols)    
    train_df = train_df.drop(*[c+'_index' for c in nominal_cols])
    # train_df = train_df.cache()
    # train_df.show(n=5,truncate=False,vertical=True)
    # print(time()-t0)


if F == 41:
    t0 = time()
    idxs = [StringIndexer(inputCol=c,outputCol=c+'_index') for c in nominal_cols]
    ohes = [OneHotEncoderEstimator(inputCols=[c+'_index'],outputCols=[c+'_numeric'],dropLast=False) for c in nominal_cols]
    idxs.extend(ohes)
    OhePipeline = Pipeline(stages=idxs)
    train_df = OhePipeline.fit(train_df).transform(train_df)
    train_df = train_df.drop(*nominal_cols)
    train_df = train_df.drop(*[c+'_index' for c in nominal_cols])
    #train_df = train_df.cache()
    train_df.show(n=5,truncate=False,vertical=True)
    #print(time()-t0)

min_max_column_udf = udf(lambda x, mi, ma: (x-mi)/(ma-mi), DoubleType())

for column in numeric_cols:    
    minimum = train_df.agg({column:'min'}).collect()[0][0]
    maximum = train_df.agg({column:'max'}).collect()[0][0]
    if (maximum - minimum) > 0 :
        train_df = train_df.withColumn(column,min_max_column_udf(train_df[column],lit(minimum),lit(maximum)))

#train_df.show(n=5,truncate=False,vertical=True)    

#train_df = train_df.cache()
#train_df.show(n=5,truncate=False,vertical=True)
#print(time()-t0)

t0 = time()
all_features = [ feature for feature in train_df.columns if feature != 'label' ]
assembler = VectorAssembler( inputCols=all_features, outputCol='features')
train_df = assembler.transform(train_df)
drop_columns = [ drop for drop in train_df.columns if drop != 'label' and drop != 'features' ]
train_df = train_df.drop(*drop_columns)
# train_df.show(n=5,truncate=False,vertical=True)
# print(time()-t0)

t0 = time()
def makeDense(v):
    return Vectors.dense(v.toArray())
makeDenseUDF = udf(makeDense,VectorUDT())

train_df = train_df.withColumn('features',makeDenseUDF(train_df.features))
df = train_df.select('features','label')
# df.show(n=5,truncate=False,vertical=True)
df = df.cache()

t0 = time()

def kNN_with_k_search(df, k_start=1, k_end=101, k_step=4):
    knn = KNNClassifier(featuresCol='features', labelCol='label', topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30)
    grid = ParamGridBuilder().addGrid(knn.k,range(k_start,k_end,k_step)).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    # BinaryClassificationEvaluator default is areaUnderROC, other option is areaUnderPR
    # evaluator.setMetricName('areaUnderROC')
    cv = CrossValidator(estimator=knn,estimatorParamMaps=grid,evaluator=evaluator,parallelism=4,numFolds=3)
    cvModel = cv.fit(df)          
    result = evaluator.evaluate(cvModel.transform(df))
    print('kNN:k',cvModel.bestModel._java_obj.getK(),result)

def kNN_with_k_fixed(df,k):
    knn = KNNClassifier(featuresCol='features', labelCol='label', topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30)
    grid = ParamGridBuilder().addGrid(knn.k,[k]).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    tts = TrainValidationSplit(estimator=knn,estimatorParamMaps=grid,evaluator=evaluator,trainRatio=0.6666)
    ttsModel = tts.fit(df)
    result = evaluator.evaluate(ttsModel.transform(df))
    print('kNN:k',k,result)

def linSVC_with_tol_iter_search(df,tol_start=0, tol_end=-9, iter_start=0, iter_end=7):
    linSVC = LinearSVC(featuresCol='features',labelCol='label')
    # funniest thing, LinearSVC expects a double for its tol parameter and an integer for iterations
    # somehow this wasn't a problem in the light_spark solution, but here removing the .0 from 10.0 will
    # result in a ClassCast Exception
    tolerances = [10.0**tol_exp for tol_exp in range(tol_start,tol_end-1,-1)]
    print(tolerances)
    iterations = [10**iter_exp for iter_exp in range(iter_start,iter_end+1,1)]
    print(iterations)
    grid = ParamGridBuilder().addGrid(linSVC.maxIter, iterations).addGrid(linSVC.tol,tolerances).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    cv = CrossValidator(estimator=linSVC,estimatorParamMaps=grid,evaluator=evaluator,parallelism=4,numFolds=3)
    cvModel = cv.fit(df)
    # print(cvModel.getEstimator())
    # print(cvModel.getEstimatorParamMaps())
    # print(cvModel.avgMetrics)    
    
    result = evaluator.evaluate(cvModel.transform(df))
    print('linSVC:tol',cvModel.bestModel._java_obj.getTol(),':maxIter',cvModel.bestModel._java_obj.getMaxIter(),cvModel.bestModel.coefficients, result)

def linSVC_with_tol_iter_fixed(df,tolerance,iterations):
    linSVC = LinearSVC(featuresCol='features',labelCol='label')
    grid = ParamGridBuilder().addGrid(linSVC.maxIter,[iterations]).addGrid(linSVC.tol,[tolerance]).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    tts = TrainValidationSplit(estimator=linSVC,estimatorParamMaps=grid,evaluator=evaluator,trainRatio=0.6666)
    ttsModel = tts.fit(df)
    result = evaluator.evaluate(ttsModel.transform(df))
    print('linSVC:tol',tolerance,':maxIter',iterations,':result',result)

def binLR_with_tol_iter_search(df,tol_start=0, tol_end=-9, iter_start=0, iter_end=7):
    binLR = LogisticRegression(featuresCol='features',labelCol='label')
    # funniest thing, LinearSVC expects a double for its tol parameter and an integer for iterations
    # somehow this wasn't a problem in the light_spark solution, but here removing the .0 from 10.0 will
    # result in a ClassCast Exception
    tolerances = [10.0**tol_exp for tol_exp in range(tol_start,tol_end-1,-1)]
    print(tolerances)
    iterations = [10**iter_exp for iter_exp in range(iter_start,iter_end+1,1)]
    print(iterations)
    grid = ParamGridBuilder().addGrid(binLR.maxIter, iterations).addGrid(binLR.tol,tolerances).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    cv = CrossValidator(estimator=binLR,estimatorParamMaps=grid,evaluator=evaluator,parallelism=4,numFolds=3)
    cvModel = cv.fit(df)
    #print(cvModel.getEstimator())
    #print(cvModel.getEstimatorParamMaps())
    #print(cvModel.avgMetrics)    
    
    result = evaluator.evaluate(cvModel.transform(df))
    print('linSVC:tol',cvModel.bestModel._java_obj.getTol(),':maxIter',cvModel.bestModel._java_obj.getMaxIter(),cvModel.bestModel.coefficients, result)

def binLR_with_tol_iter_fixed(df,tolerance,iterations):
    binLR = LogisticRegression(featuresCol='features',labelCol='label')
    grid = ParamGridBuilder().addGrid(binLR.maxIter,[iterations]).addGrid(binLR.tol,[tolerance]).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    tts = TrainValidationSplit(estimator=binLR,estimatorParamMaps=grid,evaluator=evaluator,trainRatio=0.6666)
    ttsModel = tts.fit(df)
    result = evaluator.evaluate(ttsModel.transform(df))
    print('binLR:tol',tolerance,':maxIter',iterations,'result',result)

def DTree_with_maxFeatures_maxDepth_search(df,max_depth=5,max_features=2):
    DTree = DecisionTreeClassifier(featuresCol='features',labelCol='label',impurity='gini',maxMemoryInMB=1024)
    features = list(range(2,max_features,1))
    features.extend([round(math.sqrt(F)),round(math.log2(F)),F])
    depths = list(range(1,max_depth+1,1))
    grid = ParamGridBuilder().addGrid(DTree.maxDepth,depths).addGrid(DTree.maxBins,features).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    cv = CrossValidator(estimator=DTree,estimatorParamMaps=grid,evaluator=evaluator,parallelism=4,numFolds=3)
    cvModel = cv.fit(df)
    result = evaluator.evaluate(cvModel.transform(df))
    print('DTree:maxDepth',cvModel.bestModel._java_obj.getMaxDepth(),':maxBins',cvModel.bestModel._java_obj.getMaxBins(), result)

def DTree_with_maxFeatures_maxDepth_fixed(df,max_depth,max_features):
    DTree = DecisionTreeClassifier(featuresCol='features',labelCol='label',impurity='gini',maxMemoryInMB=1024)        
    grid = ParamGridBuilder().addGrid(DTree.maxDepth,[max_depth]).addGrid(DTree.maxBins,[max_features]).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    tts = TrainValidationSplit(estimator=DTree,estimatorParamMaps=grid,evaluator=evaluator,trainRatio=0.6666)
    ttsModel = tts.fit(df)
    result = evaluator.evaluate(ttsModel.transform(df))
    print('DTree:maxDepth',max_depth,':maxBins',max_features,':result',result)

def RForest_with_maxFeatures_maxDepth_search(df,max_depth=5,max_features=2):
    RForest = RandomForestClassifier(featuresCol='features',labelCol='label',impurity='gini',maxMemoryInMB=1024)
    features = list(range(2,max_features,1))
    features.extend([round(math.sqrt(F)),round(math.log2(F)),F])
    depths = list(range(1,max_depth+1,1))
    grid = ParamGridBuilder().addGrid(RForest.maxDepth,depths).addGrid(RForest.maxBins,features).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    cv = CrossValidator(estimator=RForest,estimatorParamMaps=grid,evaluator=evaluator,parallelism=4,numFolds=3)
    cvModel = cv.fit(df)
    result = evaluator.evaluate(cvModel.transform(df))
    print('RForest:maxDepth',cvModel.bestModel._java_obj.getMaxDepth(),':maxBins',cvModel.bestModel._java_obj.getMaxBins(), result)

def RForest_with_maxFeatures_maxDepth_fixed(df,max_depth,max_features):
    RForest = DecisionTreeClassifier(featuresCol='features',labelCol='label',impurity='gini',maxMemoryInMB=1024)        
    grid = ParamGridBuilder().addGrid(RForest.maxDepth,[max_depth]).addGrid(RForest.maxBins,[max_features]).build()
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
    tts = TrainValidationSplit(estimator=RForest,estimatorParamMaps=grid,evaluator=evaluator,trainRatio=0.6666)
    ttsModel = tts.fit(df)
    result = evaluator.evaluate(ttsModel.transform(df))
    print('RForest:maxDepth',max_depth,':maxBins',max_features,':result',result)

if A == 'kNN':
    # kNN_with_k_search(df,k_start=1,k_end=51,k_step=2)    
    kNN_with_k_fixed(df,1)
elif A == 'linSVC':
    # crossed = linSVC_with_tol_iter_search(df, tol_start=0, tol_end=-9, iter_start=0, iter_end=7)
    crossed = linSVC_with_tol_iter_fixed(df,0.1,10)
elif A == 'binLR':
    # crossed = binLR_with_tol_iter_search(df, tol_start=0, tol_end=-9, iter_start=0, iter_end=7)
    crossed = binLR_with_tol_iter_fixed(df,0.001,100)
    # crossed = binLR_with_tol_iter_fixed(df,0.1,10)
    # crossed = binLR_with_tol_iter_fixed(df,0.00001,10000)
elif A == 'DTree':
    #crossed = DTree_with_maxFeatures_maxDepth_search(df, max_depth=30, max_features=F)
    #crossed = DTree_with_maxFeatures_maxDepth_fixed(df,23,14)
    #crossed = DTree_with_maxFeatures_maxDepth_fixed(df,23,14)
    crossed = DTree_with_maxFeatures_maxDepth_fixed(df,23,38)
elif A == 'RForest':
    #crossed = RForest_with_maxFeatures_maxDepth_search(df, max_depth=30, max_features=F)
    # crossed = RForest_with_maxFeatures_maxDepth_fixed(df,28,14)
    # crossed = RForest_with_maxFeatures_maxDepth_fixed(df,28,16)
    crossed = RForest_with_maxFeatures_maxDepth_fixed(df,26,37)

    
print('Total time elapsed',str(timedelta(seconds=time()-totaltime)))
print('Features',F,'Algorithm',A)    
    
'''
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4)), 122 features (F41)
Top result shows that k=1 yields the highest accuracy 1h 8min 4s runtime intel core i5 4690 @3.5GHz
1, 0.9999447171214529 (auROC)
'''

'''
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4)), 95 features (F16)
Top 4 results show that k=1 yields the highest accuracy 53min 59s runtime intel core i5 4690 @3.5GHz
1, 0.9994427576919176 (auROC)
'''

'''
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4)), 14 features (F14)
Top 4 results show that k=1 yields the highest accuracy 49min 1s runtime intel core i5 4690 @3.5GHz
1, 0.9986304780573146 (auROC)
'''

'''
Full dataset, 2/3 train, 1/3 test, k=1
F14: auPR: 0.998417467440286 1m 10s
F16: auPR: 0.999231174454344 1m 31s
F41: auPR: 0.9999181015921388 3m 19s
F14: auROC: 0.9986304780573146 57s
F16: auROC: 0.9994427576919176 1m 22s
F41: auROC: 0.9999447171214529 3m 14s
'''