# Deep Birdsnap
Fine-grained classification of bird species using Deep Learning on the [Birdsnap dataset](http://birdsnap.com/)

### Directory structure: 
```
~/birdsnap_dataset/
   deepbirdsnap/ - Our Git repo with all the code. Models, logs, etc. are gitignored. 
   bash_scripts/ - For data wrangling 
   best_weights/ - best weights for each iterations
   bottleneck/ - FC bottlenecks of data
   console_dumps/ - Keras training console output dumps and some manual logging when using 
                    custom generator with fit function. The logs you're looking for should 
		    be either here or in logs/.
      logs/ - Tensorflow logs generated as a result of callbacks. 
         old/ - archived files
   train/ - 42320 images 
   validation/ - 3000 images
   test/ - 4500 images
   removed/ - images present in original data not being used. 
      1 - cant open image, 
      5 - .gif files and 
      3 - removed to make train data size multiple of 10 for bottleneck generation
   train_small/ - small train set for debug
/data/
   `sudo mount /dev/xvdf /data` to mount
   bottlenecks_bak/ - bottleneck numpy arrays
/belly/
   `sudo mount /dev/xvdg /belly` to mount
   contains local deployment of birdsnap.com
~/CUB_2011_200/
   Experiments with the CUB dataset. These are files from birdsnap repo copied over 
   and modified to use the CUB dataset. This is not a part of the Git repo. Files 
   will be old but working. 
```

### Naming convention:
```
frozen: 	layer is not trainable
defrosted: 	some layers that were frozen are now trainable
parts:	 	regressor on bird parts
bb: 		regressor on bounding boxes
no suffix: 	bird specie classifier
whole/all: 	all layers
fc/top: 	just the fully connected layers on top
_60 or _[0-9]+: accuracy (for classification) or loss (for regression) for the experiment
bottleneck_60:  60 (or another num) represents the accuracy of network used to extract bottleneck features
2t: 		two towers
predAndWhole:	two tower arch takes predicted crop and whole image as inputs
cropped1.1:     images cropped to ground truth bb * 1.1 size
cropped_pred_scale1.1: images cropped to predicted bb * 1.1 size
```
### General stuff
- Change `exp_name` in script before running an experiment. 
- Ensure train, validation, and test paths are correctly specified in file. 
- Ensure `bottleneck_dir` points to correct dir
- Check if data augmentation in `ImageDataGenerator` is as intended.

### Image classification: 
- `source activate newbirds` - from repo root to activate the virtualenv
- `python3 extract_bottleneck_flow.py` - Extract bottleneck features (or find saved bottlenecks at `/data/bottlenecks_bak/`)
- `python3 train_top_model.py` - Train top model on bottlenecks
- `python3 defrost_train_entire_model.py` - Fine tune whole model

### Bounding box regressor: 
- `source activate newbirds`
- `python3 extract_bottleneck_flow.py` (or find saved bottlenecks at `/data/bottlenecks_bak/`)
- `python3 train_top_bb.py` - Train top model on bottlenecks
- `python3 defrost_whole_bb.py` - Fine tune whole model

### Cropped images: 
- Use `image_cropper.py` to crop. 
- Images cropped to predicted bb in folder `birdsnap_dataset/cropped_pred_scale1.1`. 
- Bottlenecks of cropped images in `/data/bottlenecks_bak/`

### Two tower image classification: 
- `source activate newbirds`
- Make sure bottlenecks of whole and cropped are in `/data/bottlenecks_bak/`.
- Train top model: `python3 train_top_2t_2fc.py` (`_2fc` = 2 FC layers)
- No defrost: both towers dont fit in GPU 

### Logs: 
- In `logs/` when using `evaluate_generator()`. 
- In `console_dumps/` when using our own generator. 
- [warning] A few files don't have the logging done properly. Always good to check if generator callback or `console_dumps` line is present in file. Else [copy buffer to file after run completes](https://ricochen.wordpress.com/2011/04/07/capture-tmux-output-the-much-less-painful-way/). 


### About class labels: 
- Validation and test sets: `get_fixed_labels(k)` generates `k` labels of each class. We're sure that valid and test data has equal images in every class ans we know this number. 
- Train images: [hacky] Each class has variable num of images. Put `bash_scripts/labels.sh` in your `train/`, run `source labels.sh > train_labels.txt` to get list of labels for use in `utils.get_labels_from_file()`. For the Birdsnap, use `deepbirdsnap/train_labels.txt`, for CUB use `CUB_2011_200/pyscripts/labels.txt`. 
- `parts_extractor.py` extracts bounding box and parts info from `parts_info.txt` and saves it to 

### Misc:
Following are the details of our AWS setup. 
- AMI: ami-1dec490b or any other with Keras, Tensorflow and CUDA preinstalled. 
- Username for AMI: ubuntu
- Instance name: `tiger`
- Availability zone: us-east-1b
- Python virtual environment: newbirds

#### Currently available storage devices:
The following EBS volumes exist as storage devices attached to `tiger`:
```
|Device	      | Mount point| Size     | AWS Volume name |
|-------------|------------|----------|-----------------|
|/dev/xvda    | /          | 150GB    | tiger-vol       |
|/dev/xvdf    | /data      |  50GB    | tiger-bak       |
|/dev/xvdg    | /belly     | 100GB    | tiger-big-paws  |
```

`/` is mounted by default (ofc). `/data` stores bottlenecks. You'll need to mount it before running experiments that use stored bottlenecks. 

Mounting a device that is available:
Mount the volume of interest (eg `xvdf` to `/data`): 
```$ sudo mount /dev/xvdf /data```

You can also [attach new EBS volumes to an instance](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html). Make sure that you create an EBS volume in the same availability zone as the instance (in our case, `us-east-1b`).
