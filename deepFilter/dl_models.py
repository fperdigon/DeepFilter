#============================================================
#
#  Deep Learning BLW Filtering
#  Deep Learning models
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================


import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, concatenate, Activation, Input

def NN_Model():

    model = Sequential()
    model.add(Dense(300, input_dim=512, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(1, activation='linear'))

    #model.add(Dense(1, activation='sigmoid'))
    return model

def deep_filter_model():
    # TODO: Make the doc

    model = Sequential()
    model.add(Conv1D(filters=128,
                     kernel_size=10,
                     activation='linear',
                     input_shape=(512, 1),
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=64,
                     kernel_size=10,
                     activation='linear',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=32,
                     kernel_size=10,
                     activation='linear',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=16,
                     kernel_size=10,
                     activation='linear',
                     strides=1,
                     padding='same'))
    model.add(Conv1D(filters=1,
                     kernel_size=10,
                     activation='linear',
                     strides=1,
                     padding='same'))

    return model


def LANLFilter_module(x, layers):
    LB = Conv1D(filters=int(layers/2),
                kernel_size=10,
                activation='linear',
                strides=1,
                padding='same')(x)

    NLB = Conv1D(filters=int(layers/2),
                 kernel_size=10,
                 activation='relu',
                 strides=1,
                 padding='same')(x)

    x = concatenate([LB, NLB])
    # x = BatchNormalization()(x)

    return x


def LANLFilter_module_v3(x, layers):
    LB1 = Conv1D(filters=int(layers / 6),
                kernel_size=5,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 6),
                kernel_size=9,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 6),
                kernel_size=15,
                activation='linear',
                strides=1,
                padding='same')(x)

    NLB1 = Conv1D(filters=int(layers / 6),
                 kernel_size=5,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 6),
                 kernel_size=9,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 6),
                 kernel_size=15,
                 activation='relu',
                 strides=1,
                 padding='same')(x)

    x = concatenate([LB1, LB2, LB3, NLB1, NLB2, NLB3])
    # x = BatchNormalization()(x)

    return x

def LANLFilter_module_v4(x, layers):
    LB1 = Conv1D(filters=int(layers / 6),
                kernel_size=5,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 6),
                kernel_size=9,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 6),
                kernel_size=15,
                dilation_rate=3,
                activation='linear',
                padding='same')(x)

    NLB1 = Conv1D(filters=int(layers / 6),
                 kernel_size=5,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 6),
                 kernel_size=9,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 6),
                 kernel_size=15,
                 dilation_rate=3,
                 activation='relu',
                 padding='same')(x)

    x = concatenate([LB1, LB2, LB3, NLB1, NLB2, NLB3])
    # x = BatchNormalization()(x)

    return x

def deep_filter_model_v2():
    # TODO: Make the doc

    input_shape = (512, 1)  # I will assuame a 256x256x1 input shape
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 128)
    tensor = LANLFilter_module(tensor, 64)
    tensor = LANLFilter_module(tensor, 32)
    tensor = LANLFilter_module(tensor, 16)
    predictions = Conv1D(filters=1,
                    kernel_size=10,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model

def deep_filter_model_v3():
    # TODO: Make the doc

    input_shape = (512, 1)  # I will assuame a 256x256x1 input shape
    input = Input(shape=input_shape)

    tensor = LANLFilter_module_v3(input, 64)
    tensor = LANLFilter_module_v3(tensor, 64)
    tensor = LANLFilter_module_v3(tensor, 32)
    tensor = LANLFilter_module_v3(tensor, 32)
    tensor = LANLFilter_module_v3(tensor, 16)
    tensor = LANLFilter_module_v3(tensor, 16)
    predictions = Conv1D(filters=1,
                    kernel_size=10,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model

def deep_filter_model_v4():
    # TODO: Make the doc

    input_shape = (512, 1)  # I will assuame a 256x256x1 input shape
    input = Input(shape=input_shape)

    tensor = LANLFilter_module_v3(input, 64)
    tensor = LANLFilter_module_v4(tensor, 64)
    tensor = LANLFilter_module_v3(tensor, 32)
    tensor = LANLFilter_module_v4(tensor, 32)
    tensor = LANLFilter_module_v3(tensor, 16)
    tensor = LANLFilter_module_v4(tensor, 16)
    predictions = Conv1D(filters=1,
                    kernel_size=10,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model

def deep_filter_model_v5():
    # TODO: Make the doc

    input_shape = (512, 1)  # I will assuame a 256x256x1 input shape
    input = Input(shape=input_shape)

    tensor = LANLFilter_module_v3(input, 64)
    tensor = LANLFilter_module_v4(tensor, 64)
    tensor = LANLFilter_module_v3(tensor, 32)
    tensor = LANLFilter_module_v4(tensor, 32)
    tensor = LANLFilter_module_v3(tensor, 16)
    tensor = LANLFilter_module_v4(tensor, 16)
    tensor = Conv1D(filters=1,
                    kernel_size=10,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)
    predictions = BatchNormalization()(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model