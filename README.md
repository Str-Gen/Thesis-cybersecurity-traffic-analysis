# Thesis-cybersecurity-traffic-analysis

This readme contains information on the structure of this repository and how to run the central solutions

## Structure

- **NSL_KDD.py, light_spark_NSL_KDD.py and heavy_spark_NSL_KDD.py: the three main implementations**
- apt2: Github link to my mirror of the Metasploit APT2 tool by Rapid7
- GeoLite2-City: a folder containing Geo-database files to be used in feature building
- NSL_KDD_Dataset: the currently used data set in two different formats (txt & csv)
- experiment-rspecs: collection of Virtual Wall experiment layouts
- ML-basics: some early learning to write ML solutions, focusing on different parts of the process
- packet_dissection.py: a Python scapy script to interpret pcap files

## Running the solutions

### NSL_KDD.py

A machine-learning solution to test categorization algorithms on the NSL_KDD dataset.
It uses Pandas to manipulate the data and scikit-learn for the classification algorithms.
The rest of the code is custom. It serves as a baseline implementation, to be run on a single machine.

#### Requirements

- Python2.7+ with dependencies

`pip install --user numpy pandas sklearn`

### light_spark_nsl_kdd.py

The ligth Spark implementation only makes use of the Spark processing engine to run the algorithms.
Data preparation and manipulation is still done with pandas and the majority of the custom code is unchanged.
This test case was built to differentiate between a solution that makes minimal use of Spark's API versus and the heavy Spark implementation that maximally uses Spark's features.

#### Requirements

- Python3.6+ with dependencies

`pip install --user pandas pyspark`

- Spark 2.3 [Spark download page](https://spark.apache.org/downloads.html)

It is advised to use your distribution's package manager to install Spark to have it update with the other installed programs.

- Spark-knn [spark-knn Github](https://github.com/saurfang/spark-knn)

spark-knn is an external spark-package, to build it yourself use the following steps

```shell
git clone https://github.com/saurfang/spark-knn.git
cd spark-knn

# Build the python part (PySpark interface for the scala implementation)
cd python
python setup.py bdist_egg

# Build the scala part (the actual algorithm)
cd ..
sbt package
```

### heavy_spark_nsl_kdd.py

The fully distributed implementation, relying on Spark-SQL to manipulate the data and using Spark-ML for the algorithms, the preprocessing steps, validation and evaluation metrics. All steps are bundled in Pipelines to create a coherent total package. 

#### Requirements

- Python3.6+ with dependency

`pip install --user pyspark`

- Spark 2.3 (same install procedure as mentioned above)
- Spark-knn external Spark-pkg (same install procedure as mentioned above)

### Script options

The scripts are built to be as uniform as possible to allow genuine comparison. The outward interface is therefore the same for all versions and currently includes:

- -F --features 14,16 or 41
- -A --algorithm knn, DTree, RForest, linSVC, binLR

### Submission to spark-submit

The spark-submit script is used to hand the python file to the Spark engine for distributed execution. An example invocation:
Obviously the paths should be changed to match your built objects. 
_Caveat spark in local model doesn't interpret spark context options like driver memory, that's why specification on the command line for spark-submit is necessary, failing to do this will almost certainly result in a heap memory shortage or garbage collection error_

```shell
SPARK_PRINT_LAUNCH_COMMAND=true spark-submit \
 --py-files spark-knn/python/dist/pyspark_knn-0.1-py3.6.egg \
 --driver-class-path spark-knn/spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-84aecdb78cb7338fb2e49254f6fdddf508d7273f.jar \
 --jars spark-knn/spark-knn-core/target/scala-2.11/spark-knn_2.11-0.0.1-84aecdb78cb7338fb2e49254f6fdddf508d7273f.jar \
 --driver-memory 12g \
 --num-executors 4 \
 light_spark_NSL_KDD.py -F 14 -A kNN
 ```

### Note about pip

Because python2 and python3 are currenlty used, it is easy to install dependencies for the wrong version. 
To find out which Python pip will be used by default, look at the endpoint of the symlink at /usr/bin/python.
If this points to python3, invoking pip will install modules for Python3.
To install modules specifically for Python2 use:

`python2 --user -m pip install [module]`