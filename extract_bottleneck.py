"""
Extract bottleneck features of Inceptionv4
Bottleneck features = activations of layer before FC layer
"""

import numpy as np
#from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from inceptionv4 import create_model
from utils import img_parts_generator
from progress.bar import Bar

img_width, img_height = 299, 299

train_data_dir = '../train'
validation_data_dir = '../validation'
test_data_dir = '../test'

nb_classes = 500
nb_parts = 2*19
nb_train_samples = 42320
nb_validation_samples = 3000
nb_test_samples = 4500

batch_size = 10

bottleneck_dir = 'bottleneck/'
exp_name = 'bottleneck_60'

def save_bottleneck_features():
    weights_file ='best_weights/defrost_everything_init_47_freeze_fixed_weights.03-0.60.hdf5' 
    model = create_model(num_classes=500, include_top=False, weights=weights_file)
    # num classes is 500 coz trained weights are with that architecture
    print(model.summary())

    for path,num in zip(['train_vsmall', 'validation', 'train', 'test'], [30,3000,42320,4500]):
        print('Saving '+path+' bottlenecks')
        generator = img_parts_generator('parts_info.txt', data_dir=path+'/', batch_size=batch_size)
        bottlenecks = None
        count = 0
        bar = Bar('Extracting ' + path, max = num//batch_size)
        for img_data, _ in generator:
            count += 1
            #print('Processing Batch: ', count)
            batch_bottleneck = model.predict(img_data, batch_size = 10)
            if bottlenecks is None:
                bottlenecks = batch_bottleneck
            else:
                bottlenecks = np.concatenate((bottlenecks, batch_bottleneck))
            bar.next()
        bar.finish()
        #bottlenecks = model.predict_generator(generator, num//batch_size, verbose=1)
        save_file = open(bottleneck_dir + exp_name + '_' + path + '.npy', 'wb')
        np.save(save_file, np.array(bottlenecks))

save_bottleneck_features()
