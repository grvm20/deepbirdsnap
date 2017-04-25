"""
Load bottlenecks with target parts and train FC layer.  
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
import time
import pickle

batch_size = 10

train_data_dir = 'train/'
validation_data_dir = 'validation/'

part_file_name = 'parts_info.txt'

nb_train_samples = 42320
nb_validation_samples = 3000
nb_test_samples = 4500
epochs = 30
exp_name = 'v3top_model_parts'
def train_top_model():

    # load top model architecture
    top_model, _, _ = create_top_model(num_classes=500) 
    top_model.compile(optimizer=optimizers.Adam(),
                loss='categorical_crossentropy', 
                metrics=['accuracy',metrics.top_k_categorical_accuracy])

    best_val_loss = 100

    f = open('console_dumps/{}.txt'.format(exp_name),'w')

    unpickled_train = pickle.load(open('cache/parts_train.p','rb'))
    unpickled_valid = pickle.load(open('cache/parts_validation.p','rb'))
    #unpickled_test = pickle.load(open('cache/parts_test.p','rb'))

    for i in range(epochs):
        time_start = time.time()
        train_generator = utils.img_parts_generator(part_file_name, train_data_dir, batch_size=5000, bottleneck_file='bottleneck/bottleneck_60_train.npy', unpickled=unpickled_train)
        val_generator = utils.img_parts_generator(part_file_name, validation_data_dir, batch_size=5000, bottleneck_file='bottleneck/bottleneck_60_validation.npy', unpickled=unpickled_valid)
        j = 0
        train_eval = []
        print('Training')
        for inp, label in train_generator:
            hist = top_model.fit(inp, label, verbose=0, batch_size=400)
            res = [hist.history['loss'][0], hist.history['acc'][0]]
            train_eval.append(res)
            #print("Epoch: {}/{} Batch (train): {}/{} train_l: {:.4f} train_acc: {:.4f}".format(i+1,epochs,j+1,42320/batch_size, res[0], res[1]))
            j += 1
        # find mean of loss and acc
        train_eval = np.mean(train_eval, axis=0)
        
        val_eval = []
        j = 0
        print('Evaluating validation set')
        for inp, label in val_generator:
            res = top_model.evaluate(inp, label, verbose=0, batch_size=400)
            #print("Epoch: {}/{} Batch (valid): {}/{} val_l: {:.4f} val_acc: {:.4f}".format(i+1,epochs,j+1,3000/batch_size, res[0], res[1]))
            val_eval.append(res)
            j += 1
            if j==5:
                break
        val_eval = np.mean(val_eval, axis=0)
        if val_eval[0] < best_val_loss:
            #print('Saving weights')
            best_val_loss = val_eval[0]
            top_model.save_weights('models/{}_{}_{:.4f}.hdf5'.format(exp_name,i+1,round(best_val_loss)))
        time_taken = time.time() - time_start
        log = 'Epoch: {}, train_l: {:.4f}, train_a: {:.4f}, val_l: {:.4f}, val_a: {:.4f}, time: {:.4f}\n'.format(i+1,train_eval[0], train_eval[1], val_eval[0],val_eval[1], time_taken)
        f.write(log)
        print(log)
    f.close()
#    history_model = top_model.fit_generator(train_generator,
#              nb_validation_samples//batch_size,
#              epochs=epochs,
#              verbose=1,
#              callbacks = [tensorboard_callback, checkpoint_callback])

    #pandas.DataFrame(history_model.history).to_csv('./history/top_model_parts.csv')
    

    #ev = top_model.evaluate(test_data,test_labels, batch_size=batch_size, verbose=1)
    #print(ev)
train_top_model()
gc.collect()
