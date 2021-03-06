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
from keras import backend as K

def top_k_arg(probs, k=10):
    """
    probs: vector of probabilities (of classes)
    returns indices of top-k probabilities
    """
    indices = np.argpartition(probs, -k)[-k:]
    return indices

def top3acc(y_true,y_pred,k=3):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)

def top5acc(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)

def dir_avail(dir_name):
    """
    Create directory if not present
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_idx2label(filename='unique_labels500.txt'):
    """
    filename: file with list of class names in sorted order
    returns: idx2label mapping (list) 0...(num_labels-1)
    """
    labels = open(filename, 'r').readlines()
    labels = [l.strip() for l in labels]
    return labels

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

def load_bottlenecks(filename):
    print('Loading bottlenecks from {}...'.format(filename))
    bottlenecks = np.load(open(filename,'rb'))
    return bottlenecks

def img_parts_generator(parts_filename, data_dir, 
        cache=True,batch_size=10, steps=None, target_dim=(299,299), 
        bottlenecks=None, unpickled=None, load_image=False, 
        load_paths=False, bb_only=False, load_orig_img=False, load_parts=True):
    """
    Return batches of (numpy image, bird parts features) for given directory
    parts_filename: 
        tab separated text file containing 42 columns of info on each row
    data_dir:
        Source directory of images, example: train/
    Unpickled:
        Pickled file path. 
        Load part features from pickled file.
        Use this over cache. 
    Cache: 
        True/False. 
        Load part features from cached file.
    Target dim: 
        scale the image to provided dim
        no scaling if None
    Batch size:
        images per batch
    Steps: 
        number of batches to yield
        yield all if None
    load_paths:
        True/False
        Return img path, part data if True
    bb_only: 
        True/False
        Return only bounding box as feature
    """
    if load_parts:
        # check if cahced features exist
        if bb_only:
            cache_prefix = 'bb_'
        else:
            cache_prefix = 'parts_'
        cache_path = 'cache/'+cache_prefix+data_dir[:-1]+'.p'
        data_dir = '../'+ data_dir
        if unpickled:
            features = unpickled['features']
            count = unpickled['count']  
        elif cache and isfile(cache_path):
            print('Loaded cached features from '+cache_path)
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
                featue_range = 42
                if bb_only:
                    feature_range = 8 # bounding boxes are indices 4-7
                else:
                    feature_range = 42 # 4 + 19*2 is num of cols in file
                    
                while i<feature_range: 
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
                    if target_dim:
                        scaled_feature = scale(point,orig_dim,target_dim=target_dim)
                    else:
                        scaled_feature = point
                    
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
    #print('Images found in '+ data_dir +': '+ str(count))
   # get files in data_dir
    filenames = [file for file in glob.glob(data_dir+'*/*', recursive=True)]
    filenames.sort()
    i = 0
    num_files = len(filenames)
    
    batch_count = 0
    while i < num_files:
        
        if batch_count == steps:
            break
        
        data_img = []
        data_orig_img = []
        data_parts = []
        data_bottleneck = []
        data_paths = []
         
        for j in range(batch_size):
            if i+j>= num_files:
                break
            img = filenames[i+j]
            if bottlenecks is not None:
                data_bottleneck.append(bottlenecks[i+j])
            if(load_orig_img):
                data_orig_img.append(get_img_array(img, target_dim=None))
            if(load_image):
                data_img.append(get_img_array(img, target_dim=target_dim))
            if load_paths:
                data_paths.append(img)
            if load_parts:
                data_parts.append(features[img])
        i += batch_size
        batch_count += 1
        data_orig_img, data_img, data_parts, data_bottleneck = np.asarray(data_orig_img), np.asarray(data_img), np.asarray(data_parts), np.asarray(data_bottleneck)
        #print(data_img.shape)
        ret_values = []
        if bottlenecks:
            ret_values.append(data_bottleneck)
        if load_image:
            ret_values.append(data_img)
        if load_orig_img:
            ret_values.append(data_orig_img)
        if load_paths:
            ret_values.append(data_paths)
        if load_parts:
            ret_values.append(data_parts)

        yield tuple(ret_values)

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

def labels_contiguous(interval, num_classes=500):
    """
    interval: number of images per class
    nb_classes: number of classes

    return onehot vector for each sample (interval*nb_classes samples assumed)
    """
    labels = []
    for i in range(num_classes):
        for j in range(interval):
            onehot = [0]*num_classes
            onehot[i] = 1
            labels.append(onehot)
    return labels

