"""
Load bottlenecks and corresponding labels for train, valid, and test data and train FC layer.  
"""
import gc
import utils
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from inceptionv4 import create_top_model
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, metrics

batch_size = 128
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

def train_top_model():
    exp_name = 'top_cropped' 

    # load bottlenecks
    print('Train')
    train_data = np.load(open('bottleneck/cropped_60_train.npy', 'rb'))
    print(train_data.shape)
    train_labels = utils.get_labels_from_file('train_labels.txt')
    print(len(train_labels))

    print('Validation')
    validation_data = np.load(open('bottleneck/cropped_60_validation.npy', 'rb'))
    print(validation_data.shape)
    val_size_per_class = 6
    validation_labels = get_fixed_labels(val_size_per_class)
    
    epochs = 20

    tensorboard_callback = TensorBoard(log_dir='./logs/'+exp_name+'/', histogram_freq=0, write_graph=True, write_images=False)
    checkpoint_callback = ModelCheckpoint('./models/'+exp_name+'.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    # load top model architecture
    top_model, x, top_model_inputs = create_top_model()
    
    top_model.compile(optimizer=optimizers.Adam(),
        loss='categorical_crossentropy', metrics=['accuracy',metrics.top_k_categorical_accuracy])

    history_model = top_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              verbose=1,
              validation_data=(validation_data, validation_labels),
              callbacks = [tensorboard_callback, checkpoint_callback])

train_top_model()
gc.collect()
