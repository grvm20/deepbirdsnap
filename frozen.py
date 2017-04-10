import os
import h5py

import time, pickle, pandas

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend
from keras import optimizers

from utils import get_labels

from inceptionv4 import create_model

nb_classes = 500
class_name = get_labels() # dict with class id (starting from 0) as key and class label as value

img_width, img_height =299, 299 

train_data_dir = '../train_small'
validation_data_dir = '../validation'
test_data_dir = '../test'

nb_train_samples = 1500#42328
nb_validation_samples = 3000
nb_test_samples = 4500

train_datagen = ImageDataGenerator(
        rescale=1./255)
#        shear_range=0.2,
#        zoom_range=0.2,
#        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=16)

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=16)

#create model
frozen_inceptionv4 = create_model(num_classes=nb_classes) 

## Freezing all layers except the last
for i in range(len(frozen_inceptionv4.layers) - 1):
    if hasattr(frozen_inceptionv4.layers[i], 'trainable'):
        frozen_inceptionv4.layers[i].trainable = False

print(frozen_inceptionv4.summary())

nb_epoch = 15

# create callback for logging
frozen_tensorboard_callback = TensorBoard(log_dir='./logs/frozen_inceptionv4/', histogram_freq=0, write_graph=True, write_images=False)
# create callback for weight save
frozen_checkpoint_callback = ModelCheckpoint('./models/frozen_inceptionv4_weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

frozen_inceptionv4.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

hist_frozen_inceptionv4 = frozen_inceptionv4.fit_generator(
        train_generator,
        samples_per_epoch = nb_train_samples,
        nb_epoch = nb_epoch,
        validation_data = validation_generator,
        nb_val_samples = nb_validation_samples,
        verbose = 1,
        initial_epoch = 0,
        callbacks=[frozen_tensorboard_callback, frozen_checkpoint_callback]
)

pandas.DataFrame(hist_frozen_convet.history).to_csv("./history/frozen_inceptionv4.csv")
