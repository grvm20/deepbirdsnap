"""
Train FC layer for bounding box regression.  
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

part_file_name = 'parts_info.txt'

nb_train_samples = 42320
nb_validation_samples = 3000
nb_test_samples = 4500
epochs = 30

exp_name = 'top_bb'
bottleneck_dir = '/data/bottlenecks_bak/'
def train_top_model():

    # load top model architecture
    top_model, _, _ = create_top_model(num_classes=4) 
    top_model.compile(optimizer=optimizers.Adam(),
               loss='mean_absolute_error', metrics=['accuracy'] )

    print(top_model.summary())
    
    bottlenecks_train = utils.load_bottlenecks(bottleneck_dir+'bottleneck_60_train.npy')
    bottlenecks_valid = utils.load_bottlenecks(bottleneck_dir+'bottleneck_60_validation.npy')

    best_val_loss = 100
    f = open('console_dumps/{}.txt'.format(exp_name),'w')

    unpickled_train = pickle.load(open('cache/bb_train.p','rb'))
    unpickled_valid = pickle.load(open('cache/bb_validation.p','rb'))

    for i in range(epochs):
        time_start = time.time()
        train_generator = utils.img_parts_generator(part_file_name, 
                'train/', batch_size=5000, 
                bottlenecks=bottlenecks_train, 
                unpickled=unpickled_train, 
                bb_only=True)
        val_generator = utils.img_parts_generator(part_file_name, 
                'validation/', batch_size=5000, 
                bottlenecks=bottlenecks_valid, 
                unpickled=unpickled_valid,
                bb_only=True)
        
        train_eval = [] # stores metrics of training iterations
#        print('Training')
        for inp, label in train_generator:
            hist = top_model.fit(inp, label, verbose=0, batch_size=400)
            res = [hist.history['loss'][0], hist.history['acc'][0]]
            train_eval.append(res)
        # find mean of loss and acc
        train_eval = np.mean(train_eval, axis=0)
        
        val_eval = [] # stores metrics of validation iterations
        j = 0
#        print('Evaluating validation set')
        for inp, label in val_generator:
            res = top_model.evaluate(inp, label, verbose=0, batch_size=400)
            #print("Epoch: {}/{} Batch (valid): {}/{} val_l: {:.4f} val_acc: {:.4f}".format(i+1,epochs,j+1,3000/batch_size, res[0], res[1]))
            val_eval.append(res)
            j += 1
            #if j==5:
            #    break
        val_eval = np.mean(val_eval, axis=0)
        
        # save weights if current loss beats best loss
        if val_eval[0] < best_val_loss:
            #print('Saving weights')
            best_val_loss = val_eval[0]
            top_model.save_weights('models/{}_{}_{:.4f}.hdf5'.format(exp_name,i+1,round(best_val_loss)))
        time_taken = time.time() - time_start
        log = 'Epoch: {}, train_l: {:.4f}, train_a: {:.4f}, val_l: {:.4f}, val_a: {:.4f}, time: {:.4f}\n'.format(i+1,
            train_eval[0], train_eval[1], val_eval[0],val_eval[1], time_taken)
        f.write(log)
        print(log)
    f.close()
train_top_model()
gc.collect()
