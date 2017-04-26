# Sys
import warnings
# Keras Core
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import SGD
# Backend
from keras import backend as K
# Utils
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file

import numpy as np

#########################################################################################
# Implements the Inception Network v4 (http://arxiv.org/pdf/1602.07261v1.pdf) in Keras. #
#########################################################################################

WEIGHTS_PATH = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'

def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1), use_bias=False, freeze=False):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    
    t = not freeze # trainable is opposite of freeze
    x = Convolution2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      trainable=t)(x)
    if t:
        x = BatchNormalization(axis=channel_axis, scale=False)(x)
    else:
        x = BatchNormalization(axis=channel_axis, scale=False, trainable=False)(x)
    x = Activation('relu')(x)
    return x

def block_inception_a(input, freeze=False):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 96, 1, 1, freeze=freeze)

    branch_1 = conv2d_bn(input, 64, 1, 1, freeze=freeze)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, freeze=freeze)

    branch_2 = conv2d_bn(input, 64, 1, 1, freeze=freeze)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3, freeze=freeze)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3, freeze=freeze)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1, freeze=freeze)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_a(input, freeze=False):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 3, 3, strides=(2,2), padding='valid', freeze=freeze)

    branch_1 = conv2d_bn(input, 192, 1, 1, freeze=freeze)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3, freeze=freeze)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, strides=(2,2), padding='valid', freeze=freeze)

    branch_2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_b(input, freeze=False):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 1, 1, freeze=freeze)

    branch_1 = conv2d_bn(input, 192, 1, 1, freeze=freeze)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7, freeze=freeze)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1, freeze=freeze)

    branch_2 = conv2d_bn(input, 192, 1, 1, freeze=freeze)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1, freeze=freeze)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7, freeze=freeze)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1, freeze=freeze)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7, freeze=freeze)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1, freeze=freeze)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_b(input, freeze=False):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 192, 1, 1, freeze=freeze)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='valid', freeze=freeze)

    branch_1 = conv2d_bn(input, 256, 1, 1, freeze=freeze)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7, freeze=freeze)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1, freeze=freeze)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, strides=(2,2), padding='valid', freeze=freeze)

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_c(input, freeze=False):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 256, 1, 1, freeze=freeze)

    branch_1 = conv2d_bn(input, 384, 1, 1, freeze=freeze)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3, freeze=freeze)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1, freeze=freeze)
    branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)


    branch_2 = conv2d_bn(input, 384, 1, 1, freeze=freeze)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1, freeze=freeze)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3, freeze=freeze)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3, freeze=freeze)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1, freeze=freeze)
    branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1, freeze=freeze)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def inception_v4_base(input, freeze_level=None):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    
    freeze_stem = False
    if freeze_level is not None and freeze_level >= 0:
        freeze_stem = True

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    net = conv2d_bn(input, 32, 3, 3, strides=(2,2), padding='valid', freeze=freeze_stem)
    net = conv2d_bn(net, 32, 3, 3, padding='valid', freeze=freeze_stem)
    net = conv2d_bn(net, 64, 3, 3, freeze=freeze_stem)

    branch_0 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)

    branch_1 = conv2d_bn(net, 96, 3, 3, strides=(2,2), padding='valid', freeze=freeze_stem)

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 64, 1, 1, freeze=freeze_stem)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, padding='valid', freeze=freeze_stem)

    branch_1 = conv2d_bn(net, 64, 1, 1, freeze=freeze_stem)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7, freeze=freeze_stem)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1, freeze=freeze_stem)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, padding='valid', freeze=freeze_stem)

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 192, 3, 3, strides=(2,2), padding='valid', freeze=freeze_stem)
    branch_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    # 35 x 35 x 384
    # 4 x Inception-A blocks

    # 35 x 35 x 384
    # Reduction-A block
    if freeze_level is not None and freeze_level >= 1:
        # create frozen A block
        for idx in range(4):
            net = block_inception_a(net, freeze=True)
        net = block_reduction_a(net, freeze=True)
    else:
        # create trainable A block
        for idx in range(4):
    	    net = block_inception_a(net)
        net = block_reduction_a(net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks

    # 17 x 17 x 1024
    # Reduction-B block
    if freeze_level is not None and freeze_level >=2:
        # create frozen B block
        for idx in range(7):
    	    net = block_inception_b(net, freeze=True)
        net = block_reduction_b(net, freeze=True)
    else:
        # create trainable B block
        for idx in range(7):
            net = block_inception_b(net)
        net = block_reduction_b(net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    if freeze_level is not None and freeze_level>=3:
        # create frozen B block
        for idx in range(3):
    	    net = block_inception_c(net, freeze=True)
    else:
        # create trainable B block
        for idx in range(3):
            net = block_inception_c(net)

    return net


def inception_v4(num_classes=500, dropout_keep_prob=0.2, weights='imagenet', include_top=False, freeze_level=None):
    '''
    Creates the inception v4 network

    Args:
    	num_classes: number of classes
    	dropout_keep_prob: float, the fraction to keep before final layer.
        include_top: whether to include top FC layers in model
        freeze_level: the level upto which the network weights are frozen
    Returns: 
    	logits: the logits outputs of the model.
    '''

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_data_format() == 'channels_first':
        inputs = Input((3, 299, 299))
    else:
        inputs = Input((299, 299, 3))

    # Make inception base
    x = inception_v4_base(inputs, freeze_level=freeze_level)


    # Final pooling and prediction

    if include_top:
        # 1 x 1 x 1536
        x = AveragePooling2D((8,8), padding='valid')(x)
        x = Dropout(dropout_keep_prob)(x)
        x = Flatten()(x)
        # 1536
        x = Dense(units=num_classes, activation='softmax')(x)

    model = Model(inputs, x, name='inception_v4')

    #print(model.summary())

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception-v4_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='9fe79d77f793fe874470d84ca6ba4a3b')
        else:
            weights_path = get_file(
                'inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='9296b46b5971573064d12e4669110969')
        model.load_weights(weights_path, by_name=True)
        print('Base weights loaded.')
        if K.backend() == 'theano':
            warnings.warn('The Theano backend is not currently supported for '
            			  'this model on Keras 2. Please use the TensorFlow  '
            			  'backend or you will get bad results!')
    return model, x, inputs


##########################
# Below are our functions#
##########################

def two_towers_top(input1=None, input2=None, weights=None, num_classes=500, activation='softmax'):
    """
    Create top model as defined by the Inceptionv4 architecture. 
    Two inputs each of 8x8x1538 (output of inceptionv4 base).
    Concat inputs. 
    Loads weights if top weights file path is specified
    """
    # input to top model is the activation after the last conv block of inception
    if input1 is None:
        input1 = Input((8,8,1536))
    if input2 is None:
        input2 = Input((8,8,1536))
    # concatenate along channel axis
    x = concatenate([input1, input2],axis=-1) 
    x = AveragePooling2D((8, 8), padding='valid')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=num_classes, activation=activation)(x)
    top_model = Model(input=[input1,input2], output=x)
    if weights: 
        top_model.load_weights(weights)
        print('Loaded top model weights')
    return top_model,x,[input1,input2]

def two_towers(tower1_weights=None, tower1_weights_output_dim=None, 
                tower2_weights=None, tower2_weights_output_dim=None,
                top_weights=None, num_classes=500, 
                activation='softmax', return_input=False,
                two_tower_weights=None):
    """
    Create model: Two conv towers, output of which concat into Dropout + FC top. 
    Weights:
        path to weight file
    freeze_level (defrost means trainable):
        None [default] = freeze nothing, whole network is trainable
        0 = freeze stem
        1 = freeze stem and A block
        2 = freeze stem and A, B blocks
        3 = freeze stem and A, B, C blocks
    include_top:
        if False, returns the base inceptionv4 model without FC layers
    top_weights:
        weights to load into top model
    activation: 
        activation function applied to output
    return_input:
        return input of model
    """
    
    # get tower 1
    tower1, input1 = create_model(include_top=False, return_input=True,
            weights=tower1_weights, weights_output_dim=tower1_weights_output_dim)
    # get tower 2
    tower2, input2 = create_model(include_top=False, return_input=True,
            weights=tower2_weightss, weights_output_dim=tower2_weights_output_dim)
    # get top
    top, x, top_inputs = two_towers_top(weights=top_weights,
                            num_classes=num_classes,
                            activation=activation)

    two_towers = Model(input=[input1,input2], 
            output=top([tower1(input1),tower2(input2)]))

    if two_tower_weights:
        two_towers.load_weights(two_tower_weights)
        
    return two_towers



def create_top_model(inputs=None, weights=None, num_classes=500, activation='softmax'):
    """
    Create top model as defined by the Inceptionv4 architecture. 
    Loads weights if top weights file path is specified
    """
    # input to top model is the activation after the last conv block of inception
    if inputs is None:
        inputs = Input((8,8,1536))
    x = AveragePooling2D((8, 8), padding='valid')(inputs)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=num_classes, activation=activation)(x)
    top_model = Model(input=inputs, output=x)
    if weights: 
        top_model.load_weights(weights)
        print('Loaded top model weights')
    return top_model,x,inputs

def create_model(weights=None, weights_output_dim=None, num_classes=500, freeze_level=None, top_weights=None, include_top=True, activation='softmax', return_input=False):
    """
    Create model with our 500 class top model. 
    Weights:
        path to weight file
    freeze_level (defrost means trainable):
        None [default] = freeze nothing, whole network is trainable
        0 = freeze stem
        1 = freeze stem and A block
        2 = freeze stem and A, B blocks
        3 = freeze stem and A, B, C blocks
    include_top:
        if False, returns the base inceptionv4 model without FC layers
    top_weights:
        weights to load into top model
    activation: 
        activation function applied to output
    """
    # get the base (inceptionv4 without top FC layers)
    base, base_x, base_inputs = inception_v4(weights=weights, include_top=False, freeze_level=freeze_level)
    print('Inceptionv4 Base loaded')
    
    if weights_output_dim: 
        # placeholder top model so that weights can be loaded to base
        top, top_x, top_inputs = create_top_model(num_classes=weights_output_dim, activation=activation)
    else:
        # this will be the actual top model that will be attached to returned model
        top, top_x, top_inputs = create_top_model(weights=top_weights, num_classes=num_classes, activation=activation)

    # create the top+base model
    defrost_inceptionv4 = fuse(base_inputs,base,top)
    
    # load weights if provided to base model (+ top) 
    if weights and weights != 'imagenet':
        defrost_inceptionv4.load_weights(weights)
        print('Weights loaded')
    
    #print(defrost_inceptionv4.layers[-2].kernel)

    if weights_output_dim is not None: # create actual top and fuse with loaded base
        # create actual top model and load top_weights into it
        top, top_x, top_inputs = create_top_model(weights=top_weights, num_classes=num_classes, activation=activation)
        # get base model of the model with loaded weights that we have till now
        base = defrost_inceptionv4.layers[-2]
        #fuse base model with actual top model
        defrost_inceptionv4 = fuse(base_inputs,base,top)
        return defrost_inceptionv4

    if not include_top:
        base_model = defrost_inceptionv4.layers[-2]
        if return_input:
            return base_model, base_inputs
        return base_model
    if return_input:
        return defrost_inceptionv4, base_inputs
    else:
        return defrost_inceptionv4

def fuse(base_inputs, base, top):
    fused = Model(input=base_inputs, output=top(base(base_inputs)))
    return fused

def test_freeze():
    """
    Test if weights are freezing
    Utilizes a simple XOR problem for this
    """
    inputs = Input((2,))
    # the trainable parameter for the layer also freezes weights
    x = Dense(units=8, activation='tanh', trainable=False)(inputs)
    #x.trainable = False # freezing weights this way for a 
    #                      functional model does not work
    x = Dense(units=1, activation='sigmoid', trainable=False)(x)
    #x.trainable = False
    model = Model(input=inputs, output=x)
    
    # freezing weights like this
    # after creating the model works
    #for l in model.layers:
    #    l.trainable = False
    
    x = np.array([[0,0], [0,1], [1,0], [1,1]], "float32")
    y = np.array([0,1,1,0],"float32")
    model.compile(optimizer=SGD(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(x,y,batch_size=1, epochs=100, verbose=1)

