from tensorflow.python.platform import gfile
import re
from PIL import Image
from os.path import isfile
import numpy as np
from progress.bar import Bar
from keras.preprocessing.image import load_img, img_to_array
import glob
import pickle
import numpy as np
def get_dim(filename):
    im = Image.open(filename)
    return im.size

def scale(point, orig_dim, target_dim=(299,299)):
    scaled_dim = (point[0]*target_dim[0]//orig_dim[0], point[1]*target_dim[1]//orig_dim[1])
    return scaled_dim

def get_img_array(path, target_dim=(299,299)):
    """
    Given path of image, returns it's numpy array
    """
    return img_to_array(load_img(path, target_size=target_dim))/255.0


def img_parts_generator(parts_filename, data_dir, batch_size=10, bottleneck_file=None, unpickled=None, load_image=False):
    """
    Get image, parts arrays for given directory of images
    example of data dir: train/
    """
    # check if cahced features exist
    cache_path = 'cache/parts_'+data_dir[:-1]+'.p'
    data_dir = '../'+ data_dir
    if unpickled:
        features = unpickled['features']
        count = unpickled['count']  
    elif isfile(cache_path):
        unpickled = pickle.load(open(cache_path,'rb'))
        features = unpickled['features']
        count = unpickled['count']    
    else:
        # load parts data from file
        with open(parts_filename,'r') as f:
            lines = f.readlines()
        
        count = 0
        print(len(lines))
        features = {} # map processed features to file path
        bar = Bar('Extracting features', max=len(lines)) # show progress bar
        for l in lines:
            #split and strip fields
            fields = l.strip().split('\t')
            fields = [x.strip() for x in fields]
            if isfile(data_dir + fields[2]):
                path = data_dir + fields[2]
                count += 1
            else:
                # continue if file is not a part of current dir
                # eg: if data_dir is validation but current sample is in train
                continue

            row = []
            i = 4 # bounding box and other part info starts from column index 4
            while i<42: # 42 columns in parts text file
                orig_dim = get_dim(path)
                field1 = fields[i]
                field2 = fields[i+1]
                # use -1 if feature does not exist
                if field1.lower() == 'null' or field2.lower() == 'null':
                    field1 = -1
                    field2 = -1
                else:
                    field1 = int(field1)
                    field2 = int(field2)
                point = (field1, field2)
                # scale the point to target img dim
                scaled_feature = scale(point,orig_dim)
                row.append(scaled_feature[0])
                row.append(scaled_feature[1])
                i += 2
            features[path] = row
            bar.next()
        bar.finish()
        # pickle this info to do it only once
        pickled = {'features':features, 'count':count}
        pickle.dump(pickled, open(cache_path,'wb'))

    # get actual img and parts
    print('Images found in '+ data_dir +': '+ str(count))
   # get files in data_dir
    filenames = [file for file in glob.glob(data_dir+'*/*', recursive=True)]
    filenames.sort()
    i = 0
    num_files = len(filenames)
    if bottleneck_file is not None:
        bottleneck = np.load(open(bottleneck_file,'rb'))
    while i < num_files:
        img = filenames[i]
        data_img = []
        data_parts = []
        data_bottleneck = []
        for j in range(batch_size):
            y = features[img]
            if bottleneck_file:
                data_bottleneck.append(bottleneck[i])
            if load_image:
                x = get_img_array(img)
                data_img.append(x)
            data_parts.append(y)
            i += 1
            if i>= num_files:
                break
        data_img, data_parts, data_bottleneck = np.asarray(data_img), np.array(data_parts), np.array(data_bottleneck)
        #print(data_img.shape)
        if bottleneck_file and load_image:
            yield data_bottleneck, data_parts, data_img
        elif load_image:
            yield data_img, data_parts
        elif bottleneck_file:
            yield data_bottleneck, data_parts
        else:
            yield data_parts
def get_labels_from_dir():
    """
    Get subdirs in a dir
    """
    image_dir = '../train'

    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    is_root_dir = True
    count = 0
    labels = {}
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        label_name = sub_dir.split('/')[2]
        labels[count] = label_name
        count += 1
    print(str(count)+' classes found')
    return labels

def get_label_mapping_from_file(filename):
    return get_labels_from_file(filename, return_mapping=True)

def get_labels_from_file(filename, return_mapping=False):
    """
    Returns values:
    0: 42323 x 500 matrix of one hot labels
    1: dict of mapping of label index to label string
    """
    labels = []
    label_name_mapping = dict()
    with open(filename,'r') as f:
        lines = f.readlines()
    lines.sort()
    print(lines[:10])
    count = -1
    prev = None
    for l in lines:
        l = l.strip()
        if prev != l:
            count += 1
            prev = l
            #print(l,count)
            label_name_mapping[count] = l
        vec = [0]*500
        vec[count] = 1
        labels.append(vec)
    print(str(count+1)+' classes found')
    print(str(len(labels))+ ' images found in '+ filename)
    if return_mapping:
        return label_name_mapping
    else:
        return labels

