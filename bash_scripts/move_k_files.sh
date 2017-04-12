# Move k files (k specified as param of head) from each subdir in 
#current dir to a specified dir (specified as param of cp)

for D in `find . -mindepth 1 -maxdepth 1 -type d`
do
echo ${D}
mkdir "../train_small/"${D}
find ${D}"/" -mindepth 1 -maxdepth 1 -type f | head -3 | xargs cp -t "../train_small/"${D}
done
