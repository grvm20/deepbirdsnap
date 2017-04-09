from tensorflow.python.platform import gfile
import re


def get_labels():
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
