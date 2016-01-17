


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

# after executing doweathclass to make data then:
#exec(open("./doweathclass_naivebayes.py").read())
#
#
from pyspark.mllib.classification import NaiveBayes

#execute model, it can go in a single pass
my_nbmodel = NaiveBayes.train(datax_rdd)

#Some info on model 
print my_nbmodel
#some checks,get some of training data and test it:
datax_col=datax_rdd.collect()   #if datax_rdd was big, use sample or take

trainset_pred =[]
for x in datax_col:
    trainset_pred.append(my_nbmodel.predict(x.features))

print trainset_pred

#to see class conditionals
#you might have to install scipy
#import scipy
#print 'Class Cond Probabilities, ie p(attr|class= 0 or 1) '
#print scipy.exp(my_nbmodel.theta)
#print scipy.exp(my_nbmodel.pi)

#get a confusion matrix
#the row is the true class label 0 or 1, columns are predicted label
#
nb_cf_mat=np.zeros([2,2])  #num of classes
for pnt in datax_col:
    predctn = my_nbmodel.predict(np.array(pnt.features))
    nb_cf_mat[pnt.label][predctn]+=1

corrcnt=0
for i in range(2):
    corrcnt+=nb_cf_mat[i][i]
nb_per_corr=corrcnt/nb_cf_mat.sum()
print 'Naive Bayes: Conf.Mat. and Per Corr'
print nb_cf_mat
print nb_per_corr

#nb_cf_mat
#array([[ 3.,  2.],
#       [ 0.,  9.]])

#0.857142857143

# for 1st train class=1 0.785714285714
