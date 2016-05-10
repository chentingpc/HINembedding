#!/bin/bash
make clean
make

data_folder=/home/doreen/Data/
result_folder=result/

./main -network $data_folder/links.txt -node2type $data_folder/node.txt -output $result_folder/emb.txt -binary 0 -size 32 -negative 5 -samples 500 -threads 11 -path_normalization 1 -rho 0.1 -sigma 0.01 -lambda 0.0

