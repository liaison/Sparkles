

#then run naivebays, then 


import numpy as np
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

#outlook,temperature,humidity,windy,play, copied from Weka's data example
rawdata=[
['sunny',85,85,'FALSE',0],
['sunny',80,90,'TRUE',0],
['overcast',83,86,'FALSE',1],
['rainy',70,96,'FALSE',1],
['rainy',68,80,'FALSE',1],
['rainy',65,70,'TRUE',0],
['overcast',64,65,'TRUE',1],
['sunny',72,95,'FALSE',0],
['sunny',69,70,'FALSE',1],
['rainy',75,80,'FALSE',1],
['sunny',75,70,'TRUE',1],
['overcast',72,90,'TRUE',1],
['overcast',81,75,'FALSE',1],
['rainy',71,91,'TRUE',0]
]

from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("PythonSpark")
sc = SparkContext(conf=conf)


from pyspark.sql import SQLContext,Row
sqlContext = SQLContext(sc)

data_df=sqlContext.createDataFrame(rawdata,
   ['outlook','temp','humid','windy','play'])

#transform categoricals into indicator variables
out2index={'sunny':[1,0,0],'overcast':[0,1,0],'rainy':[0,0,1]}


#make RDD of labeled vectors
def newrow(dfrow):
    outrow = list(out2index.get((dfrow[0])))  #get dictionary entry for outlook
    outrow.append(dfrow[1])   #temp
    outrow.append(dfrow[2])   #humidity
    if dfrow[3]=='TRUE':      #windy
        outrow.append(1)
    else:
        outrow.append(0)
    return (LabeledPoint(dfrow[4],outrow))

datax_rdd=data_df.map(newrow)

#


# ---------------- now try decision tree ------------
from pyspark.mllib.tree import DecisionTree
dt_model = DecisionTree.trainClassifier(datax_rdd,2,{},impurity='entropy',
          maxDepth=3,maxBins=32, minInstancesPerNode=2)  

#maxDepth and maxBins
#{} could be categorical feature list,
# To do regression, have no numclasses,and use trainRegression function
print dt_model.toDebugString()

#results in this:
#DecisionTreeModel classifier of depth 3 with 9 nodes
#  If (feature 1 <= 0.0)
#   If (feature 4 <= 80.0)
#    If (feature 3 <= 68.0)
#     Predict: 0.0
#    Else (feature 3 > 68.0)
#     Predict: 1.0
#   Else (feature 4 > 80.0)
#    If (feature 0 <= 0.0)
#     Predict: 0.0
#    Else (feature 0 > 0.0)
#     Predict: 0.0
#  Else (feature 1 > 0.0)
#   Predict: 1.0

#notice number of nodes are the predict (leaf nodes) and the ifs
           
#some checks,get some of training data and test it:
datax_col=datax_rdd.collect()   #if datax_rdd was big, use sample or take

#redo the conf. matrix code (it would be more efficient to pass a model)
dt_cf_mat=np.zeros([2,2])  #num of classes
for pnt in datax_col:
    predctn = dt_model.predict(np.array(pnt.features))
    dt_cf_mat[pnt.label][predctn]+=1
corrcnt=0
for i in range(2): 
    corrcnt+=dt_cf_mat[i][i]
dt_per_corr=corrcnt/dt_cf_mat.sum()
print 'Decision Tree: Conf.Mat. and Per Corr'
print dt_cf_mat
print dt_per_corr

#maxdepth 5
# print cf_mat
#[[ 5.  0.]
# [ 2.  7.]]
#>>> print per_corr
#0.857142857143


#maxdepty 3 sis same ass 5
#maxdepth 2 gives me a core dump!

