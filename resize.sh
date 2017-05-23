root="pji66/test/"
for directory in $(ls $root)
do
    for img in $(ls $root/$directory)
    do
	name=$(echo $img |  cut -d '.' -f 1)
	convert $root/$directory/$img -resize 28x28\! $root/$directory/$name.ppm
    done
done    
