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

import scipy.io as sio
import numpy as np


def Data_Preparation():

    print('Getting the Data ready ... ')

    qtdatabase = sio.loadmat('QTDatabase.mat')
    nstd = sio.loadmat('NoiseBWL.mat')


    #####################################
    # NSTD
    #####################################

    noise_channel1 = nstd['NoiseBWL']['channel1'][0, 0]
    noise_channel2 = nstd['NoiseBWL']['channel2'][0, 0]



    #####################################
    # Data split
    #####################################
    noise_channel1 = np.squeeze(noise_channel1, axis=1)
    noise_channel2 = np.squeeze(noise_channel2, axis=1)
    Full_Noise = np.concatenate((noise_channel2, noise_channel1))

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

    skip_beats = 0
    samples = 512

    for i in range(len(qtdatabase['QTDatabase']['Names'][0, 0][0])):
        signal_name = qtdatabase['QTDatabase']['Names'][0, 0][0][i][0]

        for b in qtdatabase['QTDatabase']['signals'][0, 0][i][0]:

            b_np = np.zeros(samples)
            b_sq = np.squeeze(b[0], axis=1)

            # There are beats with more than 512 samples (clould be up to 3500 samples)
            # Creating a threshold of 512 - init_padding samples max. gives a good compromise between
            # the samples amount and the discarded signals amount
            # before:
            # train: 74448  test: 13362
            # after:
            # train: 71893 test: 13306

            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1

                continue

            b_np[init_padding:b[0].shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name.split('.')[0] in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)


    sn_train = []
    sn_test = []

    noise_index = 0

    # Noise was added in a proportion from 0.2 to 2 times the ECG signal amplitude
    # Similar to
    # W. Muldrow, R.G. Mark, & Moody, G. B. (1984).
    # A noise stress test for arrhythmia detectors.
    # Computers in Cardiology, 381–384

    for s in beats_train:

        noise = noise_train[noise_index:noise_index + samples]

        beat_max_value = np.max(s) - np.min(s)

        noise_max_value = np.max(noise) - np.min(noise)

        Ase = noise_max_value / beat_max_value

        rnd = np.random.randint(low=20, high=200, size=1) / 100

        alpha = rnd / Ase

        signal_noise = s + alpha * noise

        sn_train.append(signal_noise)

        noise_index += samples

        if noise_index > (len(noise_train) - samples):
            noise_index = 0

    noise_index = 0

    for s in beats_test:

        noise = noise_train[noise_index:noise_index + samples]

        beat_max_value = np.max(s) - np.min(s)

        noise_max_value = np.max(noise) - np.min(noise)

        Ase = noise_max_value / beat_max_value

        rnd = np.random.randint(low=20, high=200, size=1) / 100

        alpha = rnd / Ase

        signal_noise = s + alpha * noise
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
