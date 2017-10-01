# Copy files from master data folder into split folder
# USAGE: create_split.sh src_dir target_dir num_imgs_per_class
# Example: bash create_split.sh images validation 100
if [ ! -d "$1/../$2" ]; then
	mkdir "$1/../$2"
	echo "CREATED $1/../$2"
fi

for D in `find $1 -mindepth 1 -maxdepth 1 -type d`
do
IFS='/' read -ra DD <<< "${D}"
LABEL=${DD[-1]}
echo ${D}
mkdir "$1/../$2/"${LABEL}
find ${D}"/" -mindepth 1 -maxdepth 1 -type f | head -n $3 | xargs mv -t"$1/../$2/"${LABEL}
done
