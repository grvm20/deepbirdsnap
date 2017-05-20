
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from evaluate import evaluate
from utils import img_parts_generator
from inceptionv4 import create_model
from scipy.misc import imsave
import numpy as np
import time
from keras import backend as K


# In[2]:

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def plot_data(img, x=None, y=None):
    implot = plt.imshow(img)
    if x is not None and y is not None:
        plt.plot(x,y,'o', marker=5)
    plt.show()


# In[3]:

weights = 'best_weights/defrost_all_cropped_77.hdf5'
model, inp = create_model(num_classes=500,
                     include_top=False, 
                     weights=weights,
                         return_input=True)
# this is the placeholder for the input images


# In[4]:

input_img = inp


# In[5]:

print(model.summary())


# In[6]:

print(model.layers[-1].name)


# In[5]:

from keras import backend as K
from utils import img_parts_generator
from KerasDeconv import DeconvNet


# In[6]:

generator = img_parts_generator('parts_info.txt', data_dir='../cropped/validation/', batch_size=10, load_parts=False, load_image=True)


# In[7]:

deconv_net = DeconvNet(model)

for img in generator:
    deconv = deconv_get_deconv(img, 
                'conv2d_139', 0, 'all')

    # postprocess and save image
    break
    #img.save('results/{}_{}_{}.png'.format(layer_name, feature_to_visualize, visualize_mode))

