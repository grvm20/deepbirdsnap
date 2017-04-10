"""
Load bottlenecks and corresponding labels for train, valid, and test data and train FC layer.  
"""
import gc
import utils
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from inceptionv4 import create_model
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

batch_size = 128
nb_classes = 500
def train_top_model():
    label_mapping = utils.get_label_mapping_from_file('train_labels.txt')
    
    train_data = np.load(open('bottleneck/bottleneck_features_train.npy', 'rb'))
    train_labels = utils.get_labels_from_file('train_labels.txt')

    validation_data = np.load(open('bottleneck/bottleneck_features_validation.npy', 'rb'))
    val_size_per_class = 6
    validation_labels = []
    for i in range(500):
        for j in range(val_size_per_class):
            onehot = [0]*500
            onehot[i] = 1
            validation_labels.append(onehot)
    
    test_data = np.load(open('bottleneck/bottleneck_features_validation.npy', 'rb'))
    test_labels = []
    test_size_per_class = 9
    for i in range(500):
        for j in range(test_size_per_class):
            onehot = [0]*500
            onehot[i] = 1
            test_labels.append(onehot)

    #create model
    frozen_inceptionv4 = create_model(num_classes=nb_classes)

    ## Freezing all layers except the last
    for i in range(len(frozen_inceptionv4.layers) - 1):
        if hasattr(frozen_inceptionv4.layers[i], 'trainable'):
                    frozen_inceptionv4.layers[i].trainable = False

    frozen_inceptionv4.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    epochs = 100

    tensorboard_callback = TensorBoard(log_dir='./logs/frozen_inceptionv4/', histogram_freq=0, write_graph=True, write_images=False)
    checkpoint_callback = ModelCheckpoint('./models/frozen_inceptionv4_weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')


    print(validation_data.shape)
    model = Sequential()
    model.add(Dense(1000, input_shape=(1536,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    optimizer = optimizers.Adam(lr=0.005)
    model.compile(optimizer=optimizer,
                                  loss='categorical_crossentropy', metrics=['accuracy'])

    history_model = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              verbose=1,
              validation_data=(validation_data, validation_labels),
              callbacks = [tensorboard_callback, checkpoint_callback])

    pandas.DataFrame(history_model.history).to_csv('./history/frozen_inceptionv4.csv')
   
train_top_model()
gc.collect()
