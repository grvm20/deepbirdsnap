from utils import get_parts
from keras.preprocessing.image import ImageDataGenerator

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
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size =10

train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size)


test_generator = test_datagen.flow_from_directory(
                 test_data_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size)

train_parts, validation_parts, test_parts = get_parts('parts_info.txt', train_generator, validation_generator, test_generator)
print(test_parts)
