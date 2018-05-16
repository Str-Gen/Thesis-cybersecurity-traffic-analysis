#! /usr/bin/python
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, min
import pyspark.sql.functions as sql
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoderEstimator, MinMaxScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
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
train_df = train_df.cache()
train_df.show(n=5,truncate=False,vertical=True)
print(time()-t0)

t0 = time()
test_df = read_dataset_typed(test_nsl_kdd_dataset_path)
test_df = mapping2DPipeline.fit(test_df).transform(test_df)
test_df = test_df.drop('labels','labels_numeric','2DAttackLabel')
test_df = test_df.withColumnRenamed('index_2DAttackLabel','label')
test_df = test_df.cache()
test_df.show(n=5,truncate=False,vertical=True)
print(time()-t0)

t0 = time()
idxs = [StringIndexer(inputCol=c,outputCol=c+'_index') for c in nominal_cols]
ohes = [OneHotEncoderEstimator(inputCols=[c+'_index'],outputCols=[c+'_numeric'],dropLast=False) for c in nominal_cols]
idxs.extend(ohes)
OhePipeline = Pipeline(stages=idxs)
train_df = OhePipeline.fit(train_df).transform(train_df)
train_df = train_df.drop(*nominal_cols)
train_df = train_df.cache()
train_df.show(n=5,truncate=False,vertical=True)
print(time()-t0)

t0 = time()
vect_numeric = [VectorAssembler(inputCols=[c],outputCol=c+'_vec') for c in numeric_cols ]
scls_nominal = [MinMaxScaler(inputCol=c,outputCol=c+'_scaled') for c in [x+'_numeric' for x in nominal_cols] ]
scls_numeric = [MinMaxScaler(inputCol=c+'_vec',outputCol=c+'_scaled') for c in numeric_cols ]
vect_numeric.extend(scls_numeric)
vect_numeric.extend(scls_nominal)
SclPipeline = Pipeline(stages=vect_numeric)
train_df = SclPipeline.fit(train_df).transform(train_df)
train_df = train_df.drop(*numeric_cols)
for x in nominal_cols:
    train_df = train_df.drop(x+'_numeric',x+'_index')
train_df = train_df.cache()
train_df.show(n=5,truncate=False,vertical=True)
print(time()-t0)

t0 = time()
all_features = [ feature for feature in train_df.columns if feature != 'label' ]
assembler = VectorAssembler( inputCols=all_features, outputCol='features')
train_df = assembler.transform(train_df)
drop_columns = [ drop for drop in train_df.columns if drop != 'label' and drop != 'features' ]
train_df = train_df.drop(*drop_columns)
train_df.show(n=5,truncate=False,vertical=True)
print(time()-t0)

t0 = time()
def makeDense(v):
    return Vectors.dense(v.toArray())
makeDenseUDF = udf(makeDense,VectorUDT())

train_df = train_df.withColumn('features',makeDenseUDF(train_df.features))
train_df_vectorized = train_df.select('features','label')
train_df.show(n=5,truncate=False,vertical=True)
print(time()-t0)
print(train_df.dtypes)

t0 = time()
knn = KNNClassifier(featuresCol='features', labelCol='label', topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30)
grid = ParamGridBuilder().addGrid(knn.k,range(1,101,4)).build()
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='label')
cv = CrossValidator(estimator=knn,estimatorParamMaps=grid,evaluator=evaluator,parallelism=4,numFolds=3)
print(train_df.count())
cvModel = cv.fit(train_df)
result = evaluator.evaluate(cvModel.transform(train_df))
print(result,'in',time()-t0)

# # def top_df_percent(df,key_col,k):
# #     num_records = df.count()
# #     k_percent_values = (df
# #                         .orderBy(col(key_col).desc())
# #                         .limit(round(num_records * k )))
# #                         # .select(min(key_col).alias('min'))
# #                         # .first()['min'])    
# #     print(k_percent_values)
# #     return df.filter(df[key_col] >= k_percent_values)

'''
Full dataset, 2/3 train, 1/3 test, 3-fold validation, k 1->97 (range(1,101,4))
Top result shows that k=1 yields the highest accuracy 1h 8min 4s runtime intel core i5 4690 @3.5GHz
1, 0.9999447171214529
'''