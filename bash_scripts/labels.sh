# Extract label for each file in current directory
# Pipe the output to file, import it in Python and use get_labels_from_file()
# from utils.py to get sorted order of labels for images.
# This is necessary for generating bottlenecks. ImageDataGenerator orders images
# in alphanumeric order of labels

for D in `find . -mindepth 1 -maxdepth 1 -type d`
do
for k in `find ${D}"/" -mindepth 1 -maxdepth 1 -type f`
do
echo ${D:2}
done
done
