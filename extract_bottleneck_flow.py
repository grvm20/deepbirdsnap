"""
Extract bottleneck features of Inceptionv4
Uses the default generator
Bottleneck features = activations of layer before FC layer
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from inceptionv4 import create_model

img_width, img_height = 299, 299

train_data_dir = '../cropped/train'
validation_data_dir = '../cropped/validation'
test_data_dir = '../cropped/test'

#nb_classes = 500
nb_parts = 2*19
nb_train_samples = 42320
nb_validation_samples = 3000
nb_test_samples = 4500

batch_size = 10

bottleneck_dir = 'bottleneck/'
exp_name = 'cropped_60'

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    model = create_model(num_classes=500, include_top=False, weights='best_weights/defrost_everything_init_47_freeze_fixed_weights.03-0.60.hdf5')

    print(model.summary())

    #validation_generator = datagen.flow_from_directory(
    #                    validation_data_dir,
    #                    target_size=(img_width, img_height),
    #                    batch_size=batch_size,
    #                    shuffle=False,
    #                    class_mode=None)
#
#    bottleneck_features_validation = model.predict_generator(
#                                    validation_generator, nb_validation_samples // batch_size, verbose=1)
#
#    np.save(open(bottleneck_dir + exp_name+'_validation.npy', 'wb'), bottleneck_features_validation)

    train_generator = datagen.flow_from_directory(train_data_dir,
                        target_size=(img_width, img_height),
                        batch_size=batch_size,
                        shuffle=False,
                        class_mode=None)
    print(nb_train_samples // batch_size)
    bottleneck_features_train = model.predict_generator(
            train_generator, nb_train_samples // batch_size, verbose=1)

    np.save(open(bottleneck_dir + exp_name + '_train.npy', 'wb'), bottleneck_features_train)
    
    
    test_generator = datagen.flow_from_directory(
                         test_data_dir,
                         target_size=(img_width, img_height),
                         batch_size=batch_size,
                         shuffle=False,
                         class_mode=None)

    
    bottleneck_features_test = model.predict_generator(
                                    test_generator, nb_test_samples // batch_size, verbose=1)
    
    np.save(open(bottleneck_dir + exp_name + '_test.npy', 'wb'), bottleneck_features_test)

save_bottleneck_features()
