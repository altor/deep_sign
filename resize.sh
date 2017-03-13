

root="GTSRB/Final_Training/Images"
for directory in $(ls $root)
do
    for img in $(ls $root/$directory)
    do
	convert $root/$directory/$img -resize 28x28\! $root/$directory/$img
    done
done    
