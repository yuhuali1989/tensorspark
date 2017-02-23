#!/usr/bin/env bash
function scptoemr(){
    scp -i ~/key/prod-data.pem -r ././../tensorspark hadoop@ec2-52-80-23-116.cn-north-1.compute.amazonaws.com.cn:~/yuhuali/
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