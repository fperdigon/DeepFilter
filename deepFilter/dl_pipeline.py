#============================================================
#
#  Deep Learning BLW Filtering
#  Deep Learning pipelines
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import losses
from sklearn.model_selection import train_test_split

import deepFilter.dl_models as models


# Custom loss SSD
def ssd_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-2)

# Custom loss SSD
def ssd_v2_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true)/(y_true*10 + 30), axis=-2) * 10 + K.sum(K.square(y_pred - y_true), axis=-2)


# Custom loss SAD
def sad_loss(y_true, y_pred):
    return K.sum(K.sqrt(K.square(y_pred - y_true)), axis=-2)

# Custom loss MAD
def mad_loss(y_true, y_pred):
    return K.max(K.square(y_pred - y_true), axis=-2)


def train_dl(Dataset, experiment):

    print('Deep Learning pipeline: Training the model for exp ' + str(experiment))

    [X_train, y_train, X_test, y_test] = Dataset

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)

    # ==================
    # LOAD THE DL MODEL
    # ==================


    if experiment == 0:
        # Vanilla CNN linear
        model = models.deep_filter_vanilla_linear()
        model_label = 'vanilla_linear'

    if experiment == 1:
        # Vanilla CNN non linear
        model = models.deep_filter_vanilla_Nlinear()
        model_label = 'vanilla_nonlinear'

    if experiment == 2:
        # Inception-like linear
        model = models.deep_filter_I_linear()
        model_label = 'I_like_linear'

    if experiment == 3:
        # Inception-like non linear
        model = models.deep_filter_I_Nlinear()
        model_label = 'I_like_Nlinear'

    if experiment == 4:
        # Inception-like linear and non linear
        model = models.deep_filter_I_LANL()
        model_label = 'I_like_LANL'

    if experiment == 5:
        # Inception-like linear and non linear dilated
        model = models.deep_filter_model_I_LANL_dilated()
        model_label = 'I_like_LANL_dilated'


    print('\n ' + model_label + '\n ')

    model.summary()

    epochs = int(1e5)  # 100000
    # epochs = 100
    batch_size = 32
    lr = 1e-3
    # lr = 1e-4
    minimum_lr = 1e-10

    model.compile(loss=ssd_v2_loss,
                  # optimizer=keras.optimizers.Adadelta(),
                  optimizer=keras.optimizers.Adam(lr=lr),
                  # optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss])

    # Keras Callbacks

    # checkpoint
    model_filepath = model_label + '_weights.best.hdf5'

    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor='ssd_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',  # on acc has to go max
                                 save_weights_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='ssd_loss',
                                  factor=0.5,
                                  min_delta=0.05,
                                  mode='min',  # on acc has to go max
                                  patience=2,
                                  min_lr=minimum_lr,
                                  verbose=1)

    early_stop = EarlyStopping(monitor="ssd_loss",  # "val_loss"
                               min_delta=0.05,
                               mode='min',  # on acc has to go max
                               patience=10,
                               verbose=1)

    tb_log_dir = './runs/' + model_label
    # if os.path.isdir(tb_log_dir): # remove the old TB dir
    #     shutil.rmtree(tb_log_dir, ignore_errors=True)

    tboard = TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                         write_graph=False, write_grads=False,
                         write_images=False, embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

    # To run the tensor board
    # tensorboard --logdir=./runs

    # GPU
    model.fit(x=X_train, y=y_train,
              validation_data=(X_val, y_val),
              batch_size=batch_size, epochs=epochs,
              verbose=1,
              callbacks=[early_stop,
                         reduce_lr,
                         checkpoint,
                         tboard])

    K.clear_session()



def test_dl(Dataset, experiment):

    print('Deep Learning pipeline: Testing the model')

    [train_set, train_set_GT, X_test, y_test] = Dataset

    batch_size = 32

    # ==================
    # LOAD THE DL MODEL
    # ==================

    if experiment == 0:
        # Vanilla CNN linear
        model = models.deep_filter_vanilla_linear()
        model_label = 'vanilla_linear'

    if experiment == 1:
        # Vanilla CNN non linear
        model = models.deep_filter_vanilla_Nlinear()
        model_label = 'vanilla_nonlinear'

    if experiment == 2:
        # Inception-like linear
        model = models.deep_filter_I_linear()
        model_label = 'I_like_linear'

    if experiment == 3:
        # Inception-like non linear
        model = models.deep_filter_I_Nlinear()
        model_label = 'I_like_Nlinear'

    if experiment == 4:
        # Inception-like linear and non linear
        model = models.deep_filter_I_LANL()
        model_label = 'I_like_LANL'

    if experiment == 5:
        # Inception-like linear and non linear dilated
        model = models.deep_filter_model_I_LANL_dilated()
        model_label = 'I_like_LANL_dilated'

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.1),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss])

    # checkpoint
    model_filepath = model_label + '_weights.best.hdf5'
    # load weights
    model.load_weights(model_filepath)

    # Test score
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

    K.clear_session()

    return [X_test, y_test, y_pred]