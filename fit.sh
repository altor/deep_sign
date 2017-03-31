#!/bin/bash

output_file="./result.png"
data_file="./result.data"
result_file="./result"
python2 src/main.py -e 10 > $result_file

l=($(grep "val_acc" $result_file | cut -d ' ' -f 13))

for i in $(seq 30)
do
    echo "$i ${l[$i]}" >> $data_file
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

rm $data_file
