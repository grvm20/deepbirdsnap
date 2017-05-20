### Directory structure: 

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

### Naming convention:

frozen: 	layer is not trainable
defrosted: 	some layers that were frozen are now trainable
parts:	 	regressor on bird parts
bb: 		regressor on bounding boxes
no suffix: 	bird specie classifier
whole/all: 	all layers
fc/top: 	just the fully connected layers on top
\_60 or \_[0-9]+: accuracy (for classification) or loss (for regression) for the experiment

### Misc:
Following are the details of our AWS setup. 

AMI: ami-1dec490b or any other with Keras, Tensorflow and CUDA preinstalled. 
Username for AMI: ubuntu
Instance name: `tiger`
Availability zone: us-east-1b
Python virtual environment: newbirds

#### Currently available storage devices:
The following EBS volumes exist as storage devices attached to `tiger`:
|Device	      | Mount point| Size     | AWS Volume name |
|-------------|------------|----------|-----------------|
|/dev/xvda    | /          | 150GB    | tiger-vol       |
|/dev/xvdf    | /data      |  50GB    | tiger-bak       |
|/dev/xvdg    | /belly     | 100GB    | tiger-big-paws  |

`/` is mounted by default (ofc). `/data` stores bottlenecks. You'll need to mount it before running experiments that use stored bottlenecks. 

Mounting a device that is available:
Mount the volume of interest (eg `xvdf` to `/data`): 
```$ sudo mount /dev/xvdf /data```

You can also [attach new EBS volumes to an instance](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html). Make sure that you create an EBS volume in the same availability zone as the instance (in our case, `us-east-1b`).
