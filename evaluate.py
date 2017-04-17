"""
Load model and evaluate it's performance on test set
"""
import gc
import utils_temp as utils
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from inceptionv4 import create_top_model
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import time
import pickle

batch_size = 10
nb_classes = 500

part_file_name = 'parts_info.txt'

train_data_dir = 'train/'
validation_data_dir = 'validation/'
test_data_dir = 'test/'

nb_train_samples = 42320
nb_validation_samples = 3000
nb_test_samples = 4500
epochs = 3
exp_name = 'top_model_parts'
def evaluate(get_labels=0):
    # load top model architecture
    top_model, _, _ = create_top_model(num_classes=38, weights='best_weights/top_model_parts.hdf5', activation=None) 
    top_model.compile(optimizer=optimizers.Adam(),
                loss='mean_absolute_error', metrics=['accuracy'])

    # load pickled parts info
    unpickled_test = pickle.load(open('cache/parts_test.p','rb'))

    time_start = time.time()
    if get_labels:
        test_generator = utils.img_parts_generator(part_file_name, test_data_dir, batch_size=get_labels, bottleneck_file='bottleneck/bottleneck_60_test.npy', unpickled=unpickled_test, load_image=True)
    else:
        test_generator = utils.img_parts_generator(part_file_name, test_data_dir, batch_size=4500, bottleneck_file='bottleneck/bottleneck_60_test.npy', unpickled=unpickled_test, load_image=False)
    if get_labels: 
        x = []
        y = []
        y_pred = []
        j = 0
        for inp, label, img in test_generator:
            
            preds = top_model.predict_on_batch(inp)
            if not x:
                x = img
                y = label
                y_pred = preds
            else:
                x = np.concatenate((x,img))
                y = np.concatenate((y,label))
                y_pred = np.concatenate((y_pred,preds))
            j+=1
            if j == 1:
                break
        return x,y,y_pred
    else:
        test_eval = []
        j = 0
        for inp, label in test_generator:
            res = top_model.evaluate(inp, label, verbose=0, batch_size=400)
            test_eval.append(res)
        test_eval = np.mean(test_eval, axis=0)
        print('Loss: {:.4f} Evaluate{:.4f}'.format(test_eval[0], test_eval[1]))
    time_taken = time.time() - time_start

if __name__ == "__main__":
    evaluate()
    #print(evaluate(get_labels=2))
