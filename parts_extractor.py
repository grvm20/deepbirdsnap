from utils import load_data
from keras.preprocessing.image import ImageDataGenerator
from inceptionv4 import create_model
import numpy as np

nb_classes = 500
#class_name = get_labels() # dict with class id (starting from 0) as key and class label as value

img_width, img_height =299, 299

train_data_dir = '../train_small'
validation_data_dir = '../validation'
test_data_dir = '../test'

nb_train_samples = 1500#42328
nb_validation_samples = 3000
nb_test_samples = 4500

train_datagen = ImageDataGenerator(
                rescale=1./255
                #shear_range=0.2,
                #zoom_range=0.2,
                #horizontal_flip=True)
                )

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size =10

train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)

validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                shuffle=False,
                class_mode=None)


test_generator = test_datagen.flow_from_directory(
                 test_data_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 shuffle=False,
                 class_mode=None)

train_img, train_parts, validation_img, validation_parts, test_img, test_parts = load_data('parts_info.txt', train_generator, validation_generator, test_generator)

numpy_data_dir = 'numpy_data/'

#np.save(open(bottleneck_dir + 'bottleneck_train_img.npy', 'wb'), train_img)
#np.save(open(bottleneck_dir + 'bottleneck_train_parts.npy', 'wb'), train_parts)
np.save(open(numpy_data_dir + 'validation_img.npy', 'wb'), validation_img)
np.save(open(numpy_data_dir + 'validation_parts.npy', 'wb'), validation_parts)
#np.save(open(bottleneck_dir + 'bottleneck_test_img.npy', 'wb'), test_img)
#np.save(open(bottleneck_dir + 'bottleneck_test_parts.npy', 'wb'), test_parts)

#model = create_model(weights='best_weights/defrost_everything_init_47_freeze_fixed_weights.03-0.60.hdf5', freeze_level=3)
#print(model.summary())
