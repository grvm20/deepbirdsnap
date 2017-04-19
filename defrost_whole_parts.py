"""
Load bottlenecks with target parts and train FC layer.  
"""
import gc
import utils_temp as utils
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from inceptionv4 import create_model
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import time
import pickle

batch_size = 32
nb_classes = 500

part_file_name = 'parts_info.txt'

train_data_dir = 'train/'
validation_data_dir = 'validation/'
test_data_dir = 'test/'

nb_train_samples = 42320
nb_validation_samples = 3000
nb_test_samples = 4500
epochs = 3
exp_name = 'defrost_all_parts'
def defrost_all_parts():

    # load top model architecture
    base_weights = 'best_weights/complete_defrost_62.hdf5'
    top_weights = 'best_weights/top_model_parts_50.hdf5'
    model = create_model(num_classes=38,activation=None,weights=base_weights, weights_output_dim=500, top_weights=top_weights) 
    model.compile(optimizer=optimizers.Adam(lr=1e-5),
                loss='mean_absolute_error', metrics=['accuracy'])
    print(model.summary())
    best_val_loss = 100

    f = open('console_dumps/{}.txt'.format(exp_name),'w')

    unpickled_train = pickle.load(open('cache/parts_train.p','rb'))
    unpickled_valid = pickle.load(open('cache/parts_validation.p','rb'))
    #unpickled_test = pickle.load(open('cache/parts_test.p','rb'))

    for i in range(epochs):
        # begin epoch
        time_start = time.time()
        # init  the generators for train and valid
        train_generator = utils.img_parts_generator(part_file_name, train_data_dir, batch_size=5000, load_image=True, unpickled=unpickled_train)
        val_generator = utils.img_parts_generator(part_file_name, validation_data_dir, batch_size=3000, load_image=True, unpickled=unpickled_valid)
        
        # j tracks batch in epoch
        j = 0
        train_eval = [] # stores results for each epoch
        for inp, label in train_generator:
            sub_epoch_start = time.time()
            hist = model.fit(inp, label, verbose=1, batch_size=batch_size)
            res = [hist.history['loss'][0], hist.history['acc'][0]]
            train_eval.append(res)
            sub_e_time = time.time() - sub_epoch_start
            print("[train] Epoch: {}/{} Batch: {}/{} train_l: {:.4f} train_acc: {:.4f} time: {:.2f}".format(i+1,epochs,j+1,42320/batch_size, res[0], res[1], sub_e_time))
            j += 1
        # find mean of loss and acc
        #train_eval = np.mean(train_eval, axis=0)
        
        val_eval = []
        j = 0
        print('Evaluating validation set')
        for inp, label in val_generator:
            res = model.evaluate(inp, label, verbose=1, batch_size=batch_size)
            print("[valid] Epoch: {}/{} Batch: {}/{} val_l: {:.4f} val_acc: {:.4f}".format(i+1,epochs,j+1,3000/batch_size, res[0], res[1]))
            val_eval.append(res)
            j += 1
            if j==5:
                break
        val_eval = np.mean(val_eval, axis=0)
        if val_eval[0] < best_val_loss:
            print('Saving weights')
            best_val_loss = val_eval[0]
            model.save_weights('models/{}_{}_{:.4f}.hdf5'.format(exp_name,i+1,round(best_val_loss)))
        time_taken = time.time() - time_start
        train_eval=[0.0,0.0]
        log = 'Epoch: {}, train_l: {:.4f}, train_a: {:.4f}, val_l: {:.4f}, val_a: {:.4f}, time: {:.4f}\n'.format(i+1,train_eval[0], train_eval[1], val_eval[0],val_eval[1], time_taken)
        f.write(log)
        print(log)
    f.close()
defrost_all_parts()
gc.collect()
