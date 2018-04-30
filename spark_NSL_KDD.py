#! /usr/bin/python
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors

from pyspark_knn.ml.classification import KNNClassifier


# This is a simple test app. Use the following command to run assuming you're in the spark-knn folder:
# spark-submit --py-files python/dist/pyspark_knn-0.1-py3.6.egg --driver-class-path spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-*.jar --jars spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-*.jar YOUR-SCRIPT.py


sc = SparkContext(appName='test')
sqlContext = SQLContext(sc)

print('Initializing')
training = sqlContext.createDataFrame([
    [Vectors.dense([0.2, 0.9]), 0.0],
    [Vectors.dense([0.2, 1.0]), 0.0],
    [Vectors.dense([0.2, 0.1]), 1.0],
    [Vectors.dense([0.2, 0.2]), 1.0],
], ['features', 'label'])

test = sqlContext.createDataFrame([
    [Vectors.dense([0.1, 0.0])],
    [Vectors.dense([0.3, 0.8])]
], ['features'])

knn = KNNClassifier(k=1, topTreeSize=1, topTreeLeafSize=1, subTreeLeafSize=1, bufferSizeSampleSize=[1, 2, 3])  # bufferSize=-1.0,
print('Params:', [p.name for p in knn.params])
print('Fitting')
model = knn.fit(training)
print('bufferSize:', model._java_obj.getBufferSize())
print('Predicting')
predictions = model.transform(test)
print('Predictions:')
for row in predictions.collect():
    print(row)