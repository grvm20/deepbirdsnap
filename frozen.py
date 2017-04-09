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

nb_classes = 500
class_name = get_labels() # dict with class id (starting from 0) as key and class label as value

# TODO: Check image dim
img_width, img_height = 150, 150

train_data_dir = '../train'
validation_data_dir = '../validation'
test_data_dir = '../test'

nb_train_samples = 42328
nb_validation_samples = 3000
nb_test_samples = 4501

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=64,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=64,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=64,
        class_mode='binary')


# TODO: create model
frozen_inception_v4 = 

# TODO: load weights

# build a classifier model to put on top of the convolutional model
#top_model = Sequential()
print(Flatten(input_shape=tf_model.output_shape[1:]))
#top_model.add(Flatten(input_shape=tf_model.output_shape[1:]))
#top_model.add(Dense(256, activation='relu'))
#top_model.add(Dropout(0.5))
#top_model.add(Dense(1, activation='sigmoid'))
#print(tf_model.summary())
#print(top_model.summary())
#tf_model.add(top_model)

nb_epoch = 15

# create callback for logging
frozen_tensorboard_callback = TensorBoard(log_dir='./logs/frozen_inceptionv4/', histogram_freq=0, write_graph=True, write_images=False)
# create callback for weight save
frozen_checkpoint_callback = ModelCheckpoint('./models/frozen_inceptionv4_weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

frozen_inceptionv4.compile(loss = 'binary_crossentropy',
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
