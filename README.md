# Sparkles
Applications for Spark

# How to Run 

* Download the "binary" version of Spark distribution. 

* Start the Spark master and slave nodes in standalone mode.
  * > ./sbin/start-all.sh

* Submit the python script to Spark master. One could get all the information on the console afterwards.
 * > ./bin/spark-submit.sh --master local[*] doweathclass_dectree.py 
