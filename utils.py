from tensorflow.python.platform import gfile
import re
from PIL import Image
from os.path import isfile
import numpy as np

def get_dim(filename):
    im = Image.open(filename)
    return im.size

def scale(point, orig_dim, target_dim=(299,299)):
    scaled_dim = (point[0]*target_dim[0]//orig_dim[0], point[1]*target_dim[1]//orig_dim[1])
    return scaled_dim

def get_parts(filename, train_gen, valid_gen, test_gen):
    with open(filename,'r') as f:
        lines = f.readlines()
    count = {'train':0, 'validation':0, 'test':0}
    features = {}
    for l in lines:
        fields = l.strip().split('\t')
        fields = [x.strip() for x in fields]
        if isfile("../train/"+fields[2]):
            path = "../train/"+fields[2]
            count['train'] += 1
        elif isfile("../validation/"+fields[2]):
            path = "../validation/"+fields[2]
            count['validation'] += 1
        elif isfile("../test/"+fields[2]):
            path = "../test/"+fields[2]
            count['test'] += 1
        else:
            continue

        #print(path, get_dim(path))
        row = []
        i = 8
        while i<42:
            orig_dim = get_dim(path)
            field1 = fields[i]
            field2 = fields[i+1]
            if field1.lower() == 'null' or field2.lower == 'null':
                field1 = -1
                field2 = -1
            else:
                field1 = int(field1)
                field2 = int(field2)
            point = (field1, field2)
            scaled_feature = scale(point,orig_dim)
            row.append(scaled_feature[0])
            row.append(scaled_feature[1])
            i += 2
        #print(fields[2])
        features[fields[2]] = row
    train = []
    valid = []
    test = []

    print(count)

    for img in train_gen.filenames:
        train.append(features[img])

    for img in valid_gen.filenames:
        valid.append(features[img])

    for img in test_gen.filenames:
        test.append(features[img])
    
    return np.array(train), np.array(valid), np.array(test)

def get_labels_from_dir():
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

