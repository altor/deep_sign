#!/bin/bash
data_file="./result.data"
result_file="./result"
nbEpochs=2

# Usage launch result_file activationFun
function launch {
    output_file=$1
    activationFun=$2

    python2 src/main.py -e $nbEpochs --activationFun $activationFun > $result_file

    l=($(grep "val_acc" $result_file | cut -d ' ' -f 13))

    for i in $(seq $nbEpochs)
    do
	echo "$i ${l[$(($i-1))]}" >> $data_file
    done

    gnuplot <<EOF
set terminal png enhanced
set output '$output_file'
set title "convergence"
set xlabel "nb epochs"
set ylabel "précision"
set grid
plot "$data_file" using 1:2 with lines
EOF

    # rm $data_file
    # rm $result_file
}

for fun in "relu" "tanh" "softmax" "elu" "softsign" "sigmoid" "hard_sigmoid" "linar"
do
    launch $fun".png" $fun
done
