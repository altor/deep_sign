#!/bin/bash
result_file="./result"
nbEpochs=100

# Usage launch result_file activationFun
function launch {
	data_file=$1
    output_file=$2
	nbConv1=$3
	nbConv2=$4
	szConv=$5
    activationFun=$6


    python2 src/main.py -e $nbEpochs --activationFun $activationFun --nbConv1 $nbConv1 --szConv1 $szConv --nbConv2 $nbConv2 --szConv2 $szConv > $result_file

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
}

for nbConv1 in 30 50 70 90
do 
	for nbConv2 in 30 50 70 90
	do
		for szConv in 4 5 7 9
		do
			for fun in "relu" "tanh" "elu" "linear"
			do
				launch $nbConv1"-"$nbConv2"-"$szConv"-"$fun".data" $nbConv1"-"$nbConv2"-"$szConv"-"$fun".png" $nbConv1 $nbConv2 $szConv $fun
			done
		done
	done
done 