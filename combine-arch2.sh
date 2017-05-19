#!/bin/bash
result_file="./result"
nbEpochs=2

# Usage launch result_file activationFun
function launch {
    data_file=$1
    nbConv1=$2
    nbConv2=$3
    szConv=$4
    activationFun=$5
    nbFull=$6
    arch=$7
    
    python2 src/main.py -e $nbEpochs --activationFun $activationFun --nbConv1 $nbConv1 --szConv1 $szConv --nbConv2 $nbConv2 --szConv2 $szConv --arch $arch --nbFull $nbFull > $result_file

    l=($(grep "val_acc" $result_file | cut -d ' ' -f 13))

    for i in $(seq $nbEpochs)
    do
	echo "$i ${l[$(($i-1))]}" >> $data_file
    done
}
nbConv1=50
nbConv2=250
szConv=4
fun="linear"
nbFull=450
arch="lenet"
for i in $(seq 50)
do
    launch "lenet$i.data" $nbConv1 $nbConv2 $szConv $fun $nbFull $arch
done

nbConv1=50
nbConv2=250
szConv=4
fun="linear"
nbFull=450
arch="arch2"
for i in $(seq 50)
do
    launch "lenet$i.data" $nbConv1 $nbConv2 $szConv $fun $nbFull $arch
done
