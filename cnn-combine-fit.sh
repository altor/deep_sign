#!/bin/bash
result_file="./CNN-result"
nbEpochs=100

# Usage launch result_file activationFun
function launch {
	data_file=$1
    output_file=$2
	nbConv1=$3
	szConv1=$4
	nbConv2=$5
	szConv2=$6
	nbConv3=$7
	szConv3=$8
    activationFun=$9


    python2 src/CNNmain.py -e $nbEpochs --activationFun $activationFun --nbConv1 $nbConv1 --szConv1 $szConv1 --nbConv2 $nbConv2 --szConv2 $szConv2 --nbConv3 $nbConv3 --szConv3 $szConv3 > $result_file

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

for nbConv1 in 90 100 110
do 
	for szConv1 in 7 9
	do
		for nbConv2 in 135 150 165
		do
			for szConv2 in 4 5 
			do
				for nbConv3 in 225 250 275
				do 
					for szConv3 in 4 5  
					do
						for fun in "tanh" "elu" "linear"
						do
							launch "CNN-"$nbConv1"-"$szConv1"-"$nbConv2"-"$szConv2"-"$nbConv3"-"$szConv3"-"$fun".data" "CNN-"$nbConv1"-"$szConv1"-"$nbConv2"-"$szConv2"-"$nbConv3"-"$szConv3"-"$fun".png" $nbConv1 $szConv1 $nbConv2 $szConv2 $nbConv3 $szConv3 $fun
						done
					done
				done
			done
		done
	done
done 