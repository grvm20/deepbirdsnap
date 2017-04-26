
# coding: utf-8

# In[16]:

from concurrent.futures import ThreadPoolExecutor
import cv2
from PIL import Image
import numpy as np
from utils import img_parts_generator
import pickle
import scipy
import os
import math
from progress.bar import Bar

def rotate_image(img, body_point, head_point, target_angle, flipped):
    head_x = head_point[0]
    head_y = head_point[1]
    body_x = body_point[0]
    body_y = body_point[1]
    slope = float(-head_y + body_y)/ (head_x - body_x)
    image_angle = math.degrees(math.atan(slope))
    if(flipped):
        image_angle = np.abs(image_angle)
    if image_angle < 0 and head_y < body_y:
        img = np.fliplr(img)
        flipped = True
    image_angle = np.abs(image_angle)

    image_center = tuple(reversed((np.array(img.shape)/2)[:-1]))
    rot_mat = cv2.getRotationMatrix2D(image_center,-image_angle+target_angle, 1.0)

    img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)

    return img, image_angle, flipped


def flip_image_if_needed(img, parts):
    beak_x = parts[6]
    left_eye_x = parts[18]
    if beak_x != -1 and left_eye_x != -1 and beak_x < left_eye_x:
        return np.fliplr(img), True
    else:
        return img, False



def get_save_path(path):
    save_img_dir = "/".join([save_path] + path[3:].split('/')[:2])
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    save_img_path = save_path + path[2:]
    return save_img_path



def crop(img, t_l_x, t_l_y, b_r_x, b_r_y):
    return img[t_l_y:b_r_y, t_l_x:b_r_x]



def resize(img, target_dim):
    return scipy.misc.imresize(img, target_dim)


# In[25]:

def get_bird_body_head_location(parts):
    body_x = []
    body_y = []
    head_x = []
    head_y = []
    for i in range(4, len(parts)):
        if parts[i] != -1:
            if i%2 == 0:
                if(i/2 in body_index):
                    body_x.append(parts[i])
                elif(i/2 in head_index):
                    head_x.append(parts[i])
                else:
                    print("Location neither bodyx nor headx: ", i)
            else:
                if(i//2 in body_index):
                    body_y.append(parts[i])
                elif(i//2 in head_index):
                    head_y.append(parts[i])
                else:
                    print("Location neither bodyy nor heady: ", i)
    return (np.mean(body_x),np.mean(body_y)) , (np.mean(head_x),np.mean(head_y))


# In[26]:


def rotate_bb_point(p_x, p_y, angle, y, x):
    
    center = x/2, y/2
    top_left_distance = math.hypot(center[0] - p_x, center[1] - p_y)
    
    ydiff = float(-p_y+center[1])
    xdiff = float(p_x-center[0])
    
    top_left_slope = ydiff/ xdiff
    top_left_angle = math.degrees(math.atan(top_left_slope))
    if ydiff>=0 and xdiff >= 0:
        top_left_angle = top_left_angle
    elif ydiff >=0:
        top_left_angle = 180 + top_left_angle
    elif xdiff >=0:
        top_left_angle = 360 + top_left_angle
    else:
        top_left_angle = 180 + top_left_angle
    top_left_angle = top_left_angle - angle
    p_x = center[0] + top_left_distance * math.cos(math.radians(top_left_angle))
    p_y = center[1] - top_left_distance * math.sin(math.radians(top_left_angle))
    return (p_x, p_y)


def align_crop(imgs, paths, parts):
    for i in range(len(imgs)):
        
        img = imgs[i]
        path = paths[i]
        
        t_l_x = parts[i][0]
        t_l_y = parts[i][1]
        b_r_x = parts[i][2]
        b_r_y = parts[i][3]
        
        body_point, head_point = get_bird_body_head_location(parts[i])
        
        img, flipped = flip_image_if_needed(img, parts[i])
        img, image_angle, flipped = rotate_image(img, body_point, head_point, target_angle, flipped)
        if flipped:
            y, x, rgb = img.shape
            temp = t_l_x
            t_l_x = x - b_r_x
            b_r_x = x - temp
        t_l = rotate_bb_point(t_l_x, t_l_y, -target_angle + image_angle, img.shape[0], img.shape[1])
        b_l = rotate_bb_point(t_l_x, b_r_y, -target_angle + image_angle, img.shape[0], img.shape[1])
        t_r = rotate_bb_point(b_r_x, t_l_y, -target_angle + image_angle, img.shape[0], img.shape[1])
        b_r = rotate_bb_point(b_r_x, b_r_y, -target_angle + image_angle, img.shape[0], img.shape[1])
        
        
        
        t_l_x = int(max(min(t_l[0], b_l[0]), 0))
        t_l_y = int(max(min(t_l[1], t_r[1]), 0))
        b_r_x = int(min(max(t_r[0], b_r[0]), img.shape[1]))
        b_r_y = int(min(max(b_l[1], b_r[1]), img.shape[0]))
        
        img = crop(img, t_l_x, t_l_y, b_r_x, b_r_y)
        
        resized_cropped_img = resize(img, (299, 299))
        
        save_img_path = get_save_path(path)
        
        image = Image.fromarray(resized_cropped_img)
        image.save(save_img_path)
        bar.next()

part_file_name = 'parts_info.txt'
data_dir = 'validation/'
batch_size = 1
steps=4
target_dim=None
cache=False
save_path = '../cropped_aligned'
target_angle = 50



generator = img_parts_generator(part_file_name, data_dir, batch_size=20, load_image=True, target_dim=target_dim, cache=False, load_paths=True, load_parts=True, bb_only=False)


body_index = set([2, 4, 5, 10, 11, 15, 16, 17])
head_index = set([3, 6, 7, 8, 9, 12, 13, 14, 18])

bar = Bar('Cropping aligned image', max=3000)
with ThreadPoolExecutor(max_workers=100) as executor:

    for imgs, paths, parts in generator:
        executor.submit(align_crop, imgs, paths, parts)

bar.finish()
