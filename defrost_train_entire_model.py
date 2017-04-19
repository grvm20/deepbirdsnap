"""
Defrost whole model with pre-trained FC.  
"""
import gc
import utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from inceptionv4 import create_model 
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras import metrics

batch_size = 8
nb_classes = 500

def get_fixed_labels(interval, nb_classes=500):
    """
    interval: number of images per class
    nb_classes: number of classes

    return onehot vector for each sample (interval*nb_classes samples assumed)
    """
    labels = []
    for i in range(nb_classes):
        for j in range(interval):
            onehot = [0]*500
            onehot[i] = 1
            labels.append(onehot)
    return labels

def defrost_train():
    nb_classes = 500

    img_width, img_height =299, 299

    train_data_dir = '../cropped/train'
    validation_data_dir = '../cropped/validation'
    test_data_dir = '../cropped/test'

    nb_train_samples = 42320
    nb_validation_samples = 3000
    nb_test_samples = 4500

    data_aug = False

    if data_aug:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    nb_epoch = 5
    batch_size = 16
    exp_name = 'defrost_all_cropped'
    
    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)
    
    validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

    #test_generator = test_datagen.flow_from_directory(
    #test_data_dir,
    #target_size=(img_width, img_height),
    #batch_size=batch_size)

    tensorboard_callback = TensorBoard(log_dir='./logs/'+exp_name+'/', histogram_freq=0, write_graph=True, write_images=False)
    checkpoint_callback = ModelCheckpoint('./models/'+exp_name+'_weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    # load top model architecture
    base_weights = 'best_weights/defrost_everything_init_47_freeze_fixed_weights.03-0.60.hdf5'
    top_weights = 'best_weights/top_cropped_69.hdf5'
    defrost_model = create_model(freeze_level=None, weights_output_dim=500, top_weights=top_weights, weights=base_weights)
    
    defrost_model.compile(optimizer=optimizers.Adam(lr=1e-5), #tried 6 zeros
        loss='categorical_crossentropy', metrics=['accuracy', metrics.top_k_categorical_accuracy])
    print(defrost_model.summary())
    
    hist_model = defrost_model.fit_generator(
        train_generator,
        nb_train_samples//batch_size,
        epochs = nb_epoch,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples//batch_size,
        verbose = 1,
        initial_epoch = 0,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )
    
    ev_validation = defrost_model.evaluate_generator(validation_generator,nb_validation_samples//batch_size)
    print(ev_validation)
    #ev_test = defrost_model.evaluate(test_data,test_labels, batch_size=batch_size, verbose=1)
    #print(ev_test)

defrost_train()
gc.collect()
