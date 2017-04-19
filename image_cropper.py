"""
Crop image according to provided bounding box, rescale to provided dimension
"""
from PIL import Image
import numpy as np
from utils_temp import img_parts_generator
import pickle
import scipy
import os
from progress.bar import Bar

def get_save_path(path):
    # get path of destination from source file path
    save_img_dir = "/".join([save_path] + path[3:].split('/')[:2])
    print(save_img_dir)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    save_img_path = save_path + path[2:]
    return save_img_path

def crop(img, t_l_x, t_l_y, b_r_x, b_r_y):
    return img[t_l_y:b_r_y, t_l_x:b_r_x]


def resize(img, target_dim):
    return scipy.misc.imresize(img, target_dim)

part_file_name = 'parts_info.txt'
validation_data_dir = 'validation/'
batch_size = 100
target_dim=None
cache=False
save_path = '../cropped'


for direc,num in zip(['validation/','train/','test/'],[3000,42320,4500]):
    gen = img_parts_generator(part_file_name, direc, batch_size=batch_size, load_image=True, target_dim=target_dim, cache=False, load_paths=True)
    bar = Bar('Cropping: '+direc[:-1],max=num)
    for imgs, parts, paths in gen:
        for i in range(len(imgs)):
            img = imgs[i]
            path = paths[i]
           
            # get bounding boxes
            t_l_x = parts[i][0]
            t_l_y = parts[i][1]
            b_r_x = parts[i][2]
            b_r_y = parts[i][3]
            
            img = crop(img, t_l_x, t_l_y, b_r_x, b_r_y)
            
            img = resize(img, (299, 299))
            
            save_img_path = get_save_path(path)
            
            img = Image.fromarray(img)
            img.save(save_img_path)
            bar.next()
    bar.finish()
