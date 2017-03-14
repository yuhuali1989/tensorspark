#!/usr/bin/env bash
function scptoemr(){
    scp -i ~/key/prod-data.pem -r ././../tensorspark hadoop@ec2-54-223-189-86.cn-north-1.compute.amazonaws.com.cn:~/
}
function scppytoemr(){
    scp -i ~/key/prod-data.pem -r *.py hadoop@ec2-54-223-189-86.cn-north-1.compute.amazonaws.com.cn:~/tensorspark
}
function submit(){
    zip pyfiles.zip ./parameterwebsocketclient.py ./parameterservermodel.py ./mnistcnn.py ./mnistdnn.py ./moleculardnn.py ./higgsdnn.py

    spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue default \
    --num-executors 2 \
    --driver-memory 1g \
    --executor-memory 1g \
    --executor-cores 2 \
    --py-files ./pyfiles.zip \
    ./tensorspark.py

}
function putdata(){
    hdfs dfs -mkdir -p /data/ml/tensorspark/
    hdfs dfs -put -f ./tiny_mnist_train.csv /data/ml/tensorspark/
    hdfs dfs -put -f ./mnist_train.csv /data/ml/tensorspark/
    hdfs dfs -put -f ./tiny_mnist_test.csv /data/ml/tensorspark/
    hdfs dfs -put -f ./mnist_test.csv /data/ml/tensorspark/
    sudo mkdir -p /var/log/tests/tensorspark/
}
$*