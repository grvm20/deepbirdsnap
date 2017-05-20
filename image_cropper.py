"""
Crop image according to provided bounding box, rescale to provided dimension
"""
from PIL import Image
import numpy as np
from utils import img_parts_generator
import pickle
import scipy
import os
from progress.bar import Bar
from inceptionv4 import create_model
def get_save_path(path, save_path):
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

def scale(point, target_dim, orig_dim=(299, 299)):
    dim1 = max(point[0]*target_dim[1]//orig_dim[0], 0)
    dim2 = max(point[1]*target_dim[0]//orig_dim[1], 0)
    scaled_dim = (dim1, dim2)
    return scaled_dim

def scale_bounding_box(t_l, b_r, orig_image_dim, scale=1.0):
    length = (b_r[1] - t_l[1]) * (scale-1)
    width = (b_r[0] - t_l[0]) * (scale-1)
    t_l_0 = max(t_l[0] - width/2, 0)
    t_l_1 = max(t_l[1] - length/2, 0)
    b_r_0 = min(b_r[0] + width/2, orig_image_dim[1])
    b_r_1 = min(b_r[1] + length/2, orig_image_dim[0])
    return (t_l_0, t_l_1), (b_r_0, b_r_1)

def crop():

    part_file_name = 'parts_info.txt'
    #validation_data_dir = 'validation/'
    batch_size = 100
    target_dim=(299, 299)
    cache=False
    save_path = '../cropped_pred_scale1.1'

    model = create_model(num_classes=4, weights='best_weights/defrost_all_bb_8.hdf5', activation=None)
    print(model.summary())
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    direcs = ['test/', 'train/']
    nums = [4500, 42320]
    #direcs = ['validation/']
    #nums = [3000]
    for direc,num in zip(direcs,nums):
        gen = img_parts_generator(part_file_name, direc, batch_size=batch_size, load_image=True, target_dim=target_dim, cache=False, load_paths=True, load_orig_img=True, bb_only=True)
        bar = Bar('Cropping: '+direc[:-1],max=num)
        for imgs, orig_imgs, paths, parts in gen:
            preds = model.predict(imgs,batch_size=100,verbose=1)
            for i in range(len(imgs)):
                img = imgs[i]
                orig_img = orig_imgs[i]
                path = paths[i]

                ## Rescaling predicted points to original dimensions
                t_l_point = (preds[i][0], preds[i][1])
                b_r_point = (preds[i][2], preds[i][3])
                #print("Orig: ", parts[i])
                #print("Predicted:", t_l_point, b_r_point)
                b_r_point = scale(b_r_point, orig_img.shape)
                t_l_point = scale(t_l_point, orig_img.shape)
                
                t_l_point, b_r_point = scale_bounding_box(t_l_point, b_r_point, orig_img.shape, scale=1.1)
                # get bounding boxes
                t_l_x = int(t_l_point[0])
                t_l_y = int(t_l_point[1])
                b_r_x = int(b_r_point[0])
                b_r_y = int(b_r_point[1])
                if(b_r_y > t_l_y and b_r_x>t_l_x):
                    img = crop(orig_img, t_l_x, t_l_y, b_r_x, b_r_y)
                else:
                    img = orig_img
                try:
                    img = resize(img, (299, 299))
                except ValueError:
                    print(img.shape)
                    print(orig_img.shape)
                    print(t_l_x, b_r_x, t_l_y, b_r_y)
                    print(t_l_point, b_r_point)
                    a = orig_img[t_l_y:b_r_y, :]
                    print(a.shape)
                    raise ValueError("Error")
                save_img_path = get_save_path(path, save_path)
                
                img = Image.fromarray(img)
                img.save(save_img_path)
                bar.next()
            #break
        bar.finish()
if __name__ == "__main__":
    crop()
