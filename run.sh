#!/usr/bin/env bash
function scptoemr(){
    scp -i ~/key/prod-data.pem -r ././../tensorspark hadoop@ec2-54-223-22-239.cn-north-1.compute.amazonaws.com.cn:~/
}
function submit(){
    zip pyfiles.zip ./parameterwebsocketclient.py ./parameterservermodel.py ./mnistcnn.py ./mnistdnn.py ./moleculardnn.py ./higgsdnn.py

    spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --queue default \
    --num-executors 2 \
    --driver-memory 2g \
    --executor-memory 2g \
    --executor-cores 2 \
    --py-files ./pyfiles.zip \
    ./tensorspark.py

}
function putdata(){
    hdfs dfs -mkdir -p /data/ml/tensorspark/
    hdfs dfs -put ./tiny_mnist_train.csv /data/ml/tensorspark/
}
$*