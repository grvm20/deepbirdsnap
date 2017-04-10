from tensorflow.python.platform import gfile
import re


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
