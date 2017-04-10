import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from inceptionv4 import create_model

img_width, img_height = 299, 299

train_data_dir = '../train'
validation_data_dir = '../validation'
test_data_dir = '../test'

nb_classes = 500

nb_train_samples = 42320
nb_validation_samples = 3000
nb_test_samples = 4500

batch_size = 10

bottleneck_dir = 'bottleneck/'

def save_bottlebeck_features():
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    model, x, inputs = create_model(num_classes=nb_classes, include_top=False, weights='imagenet')
    """
    validation_generator = test_datagen.flow_from_directory(
                        validation_data_dir,
                        target_size=(img_width, img_height),
                        batch_size=batch_size,
                        shuffle=False,
                        class_mode=None)

    bottleneck_features_validation = model.predict_generator(
                                    validation_generator, nb_validation_samples // batch_size)

    np.save(open(bottleneck_dir + 'bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

    
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                        target_size=(img_width, img_height),
                        batch_size=batch_size,
                        shuffle=False,
                        class_mode=None)
    print(nb_train_samples // batch_size)
    bottleneck_features_train = model.predict_generator(
            train_generator, nb_train_samples // batch_size)

    np.save(open(bottleneck_dir + 'bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    """
    test_generator = test_datagen.flow_from_directory(
                         test_data_dir,
                         target_size=(img_width, img_height),
                         batch_size=batch_size,
                         shuffle=False,
                         class_mode=None)

    
    bottleneck_features_test = model.predict_generator(
                                    test_generator, nb_test_samples // batch_size)
    
    np.save(open(bottleneck_dir + 'bottleneck_features_test.npy', 'wb'), bottleneck_features_test)

