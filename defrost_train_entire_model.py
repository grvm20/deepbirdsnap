"""
Load bottlenecks and corresponding labels for train, valid, and test data and train FC layer.  
"""
import gc
import utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from inceptionv4 import create_defrost_model,create_model_with_weights 
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

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

    train_data_dir = '../train'
    validation_data_dir = '../validation'
    test_data_dir = '../test'

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
    batch_size = 8
    
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

    tensorboard_callback = TensorBoard(log_dir='./logs/defrost/', histogram_freq=0, write_graph=True, write_images=False)
    checkpoint_callback = ModelCheckpoint('./models/defrost_weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    # load top model architecture
    defrost_model = create_model_with_weights('best_weights/defrost_weights_train_c_loaded_fc.hdf5', freeze_level=2)
    
    defrost_model.compile(optimizer=optimizers.Adam(lr=1e-5), #tried 6 zeros
        loss='categorical_crossentropy', metrics=['accuracy'])
    """
    history_model = defrost_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              verbose=1,
              validation_data=(validation_data, validation_labels),
              callbacks = [tensorboard_callback, checkpoint_callback])
    """
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

    pandas.DataFrame(hist_model.history).to_csv('./history/defrost_model.csv')
    
    #ev_validation = defrost_model.evaluate_generator(validation_generator,nb_validation_samples//batch_size)
    #print(ev_validation)
    #ev_test = defrost_model.evaluate(test_data,test_labels, batch_size=batch_size, verbose=1)
    #print(ev_test)
defrost_train()
gc.collect()
