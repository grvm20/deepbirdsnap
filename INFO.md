#### Directory structure: 

birdsnap_dataset/
	deepbirdsnap/ - Our Git repo with all the code. Models, logs, etc. are gitignored. 
		bash_scripts/ - For data wrangling 
		best_weights/ - best weights for each iterations
		bottleneck/ - FC bottlenecks of data
		console_dumps/ - Keras training console output dumps
		logs/ - Tensorflow logs generated as a result of callbacks
		old/ - archived files
	train/ - 42320 images 
	validation/ - 3000 images
	test/ - 4500 images
	removed/ - images present in original data not being used. 1 - cant open image, 5 - .gif files and 3 - removed to make train data size multiple of 10 for bottleneck generation

#### Naming convention:

frozen: 	layer is not trainable
defrosted: 	some layers that were frozen are now trainable
parts:	 	regressor on bird parts
bb: 		regressor on bounding boxes
no suffix: 	bird specie classifier
whole/all: 	all layers
fc/top: 	just the fully connected layers on top
\_60 or \_[0-9]+: accuracy (for classification) or loss (for regression) for the experiment

#### Misc:
Username for AMI: ubuntu

Python virtual environment: newbirds

Mounting Backup EBS volume:
/data/ is mount point for a backup 50GB EBS volume, mainly for storing bottlenecks
You must mount the volume before accessing it.
Get the volume name
```$ lsblk```
It should be `xvdf`
Mount the volume: 
```$ sudo mount /dev/xvdf /data```


