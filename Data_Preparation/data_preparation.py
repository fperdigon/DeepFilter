#============================================================
#
#  Deep Learning BLW Filtering
#  Data preparation
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import numpy as np
import _pickle as pickle
from Data_Preparation import Prepare_QTDatabase, Prepare_NSTDB

def Data_Preparation():

    print('Getting the Data ready ... ')

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    Prepare_QTDatabase.prepare()
    Prepare_NSTDB.prepare()

    # Load QT Database
    with open('data/QTDatabase.pkl', 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open('data/NoiseBWL.pkl', 'rb') as input:
        nstd = pickle.load(input)

    #####################################
    # NSTDB
    #####################################

    noise_channel1 = nstd[:, 0]
    noise_channel2 = nstd[:, 1]


    #####################################
    # Data split
    #####################################

    noise_test = np.concatenate(
        (noise_channel1[0:int(noise_channel1.shape[0] * 0.13)], noise_channel2[0:int(noise_channel2.shape[0] * 0.13)]))
    noise_train = np.concatenate((noise_channel1[int(noise_channel1.shape[0] * 0.13):-1],
                                  noise_channel2[int(noise_channel2.shape[0] * 0.13):-1]))

    #####################################
    # QTDatabase
    #####################################

    beats_train = []
    beats_test = []

    # QTDatabese signals Dataset splitting. Considering the following link
    # https://www.physionet.org/physiobank/database/qtdb/doc/node3.html
    #  Distribution of the 105 records according to the original Database.
    #  | MIT-BIH | MIT-BIH |   MIT-BIH  |  MIT-BIH  | ESC | MIT-BIH | Sudden |
    #  | Arrhyt. |  ST DB  | Sup. Vent. | Long Term | STT | NSR DB	| Death  |
    #  |   15    |   6	   |     13     |     4     | 33  |  10	    |  24    |
    #
    # The two random signals of each pathology will be keep for testing set.
    # The following list was used
    # https://www.physionet.org/physiobank/database/qtdb/doc/node4.html
    # Selected test signal amount (14) represent ~13 % of the total

    test_set = ['sel123',  # Record from MIT-BIH Arrhythmia Database
                'sel233',  # Record from MIT-BIH Arrhythmia Database

                'sel302',  # Record from MIT-BIH ST Change Database
                'sel307',  # Record from MIT-BIH ST Change Database

                'sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database

                'sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database

                'sele0106',  # Record from European ST-T Database
                'sele0121',  # Record from European ST-T Database

                'sel32',  # Record from ``sudden death'' patients from BIH
                'sel49',  # Record from ``sudden death'' patients from BIH

                'sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'sel15814',  # Record from MIT-BIH Long-Term ECG Database
                ]


    # Creating the train and test dataset, each datapoint has 512 samples and is zero padded
    # beats bigger that 512 samples are discarded to avoid wrong split beats ans to reduce
    # computation.
    skip_beats = 0
    samples = 512
    qtdb_keys = list(qtdb.keys())

    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]

        for b in qtdb[signal_name]:

            b_np = np.zeros(samples)
            b_sq = np.array(b)

            # There are beats with more than 512 samples (could be up to 3500 samples)
            # Creating a threshold of 512 - init_padding samples max. gives a good compromise between
            # the samples amount and the discarded signals amount
            # before:
            # train: 74448  test: 13362
            # after:
            # train: 71893 test: 13306  (discarded train: ~4k datapoints test: ~50)

            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue

            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)


    # Noise was added in a proportion from 0.2 to 2 times the ECG signal amplitude
    # Similar to
    # W. Muldrow, R.G. Mark, & Moody, G. B. (1984).
    # A noise stress test for arrhythmia detectors.
    # Computers in Cardiology, 381–384

    sn_train = []
    sn_test = []

    noise_index = 0

    # Adding noise to train
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    for i in range(len(beats_train)):
        noise = noise_train[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / Ase
        signal_noise = beats_train[i] + alpha * noise
        sn_train.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_train) - samples):
            noise_index = 0

    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    for i in range(len(beats_test)):
        noise = noise_test[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_test[i]) - np.min(beats_test[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_test[i] / Ase
        signal_noise = beats_test[i] + alpha * noise
        sn_test.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_test) - samples):
            noise_index = 0


    X_train = np.array(sn_train)
    y_train = np.array(beats_train)

    X_test = np.array(sn_test)
    y_test = np.array(beats_test)

    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)


    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset
