# -*- coding: utf-8 -*-

#============================================================
#
#  Deep Learning BLW Filtering
#  Main
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import _pickle as pickle
from datetime import datetime
import time
import numpy as np

from utils.metrics import MAD, SSD, PRD, COS_SIM
from utils import visualization as vs
from Data_Preparation import data_preparation as dp

from digitalFilters.dfilters import FIR_test_Dataset, IIR_test_Dataset
from deepFilter.dl_pipeline import train_dl, test_dl


if __name__ == "__main__":

    dl_experiments = [
                      'DRNN',
                      'FCN-DAE',
                      'Vanilla L',
                      'Vanilla NL',
                      'Multibranch LANL',
                      'Multibranch LANLD'
                      ]

    noise_versions = [1, 2]
    for nv in noise_versions:
        # Data_Preparation() function assumes that QT database and Noise Stress Test Database are uncompresed
        # inside a folder called data

        Dataset = dp.Data_Preparation(noise_version=nv)

        # Save dataset
        with open('data/dataset_nv' + str(nv) + '.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(Dataset, output)
        print('Dataset saved')

        # Load dataset
        with open('data/dataset_nv' + str(nv) + '.pkl', 'rb') as input:
            Dataset = pickle.load(input)


        train_time_list = []
        test_time_list = []

        for experiment in range(len(dl_experiments)):
            start_train = datetime.now()
            train_dl(Dataset, dl_experiments[experiment])
            end_train = datetime.now()
            train_time_list.append(end_train - start_train)

            start_test = datetime.now()
            [X_test, y_test, y_pred] = test_dl(Dataset, dl_experiments[experiment])
            end_test = datetime.now()
            test_time_list.append(end_test - start_test)

            test_results = [X_test, y_test, y_pred]

            # Save Results
            with open('test_results_' + dl_experiments[experiment] + '_nv' + str(nv) + '.pkl', 'wb') as output:  # Overwrites any existing file.
                pickle.dump(test_results, output)
            print('Results from experiment ' + dl_experiments[experiment] + '_nv' + str(nv) + ' saved')

            time.sleep(60)

        # Classical Filters

        # FIR
        start_test = datetime.now()
        [X_test_f, y_test_f, y_filter] = FIR_test_Dataset(Dataset)
        end_test = datetime.now()
        train_time_list.append(0)
        test_time_list.append(end_test - start_test)

        test_results_FIR = [X_test_f, y_test_f, y_filter]

        # Save FIR filter results
        with open('test_results_FIR_nv' + str(nv) + '.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(test_results_FIR, output)
        print('Results from experiment FIR filter nv ' + str(nv) + ' saved')

        # IIR
        start_test = datetime.now()
        [X_test_f, y_test_f, y_filter] = IIR_test_Dataset(Dataset)
        end_test = datetime.now()
        train_time_list.append(0)
        test_time_list.append(end_test - start_test)

        test_results_IIR = [X_test_f, y_test_f, y_filter]

        # Save IIR filter results
        with open('test_results_IIR_nv' + str(nv) + '.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(test_results_IIR, output)
        print('Results from experiment IIR filter nv ' + str(nv) + ' saved')

        # Saving timing list
        timing = [train_time_list, test_time_list]
        with open('timing_nv' + str(nv) + '.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(timing, output)
        print('Timing nv ' + str(nv) + ' saved')


    ####### LOAD EXPERIMENTS #######

    # Load timing
    with open('timing_nv1.pkl', 'rb') as input:
        timing_nv1 = pickle.load(input)
        [train_time_list_nv1, test_time_list_nv1] = timing_nv1

    with open('timing_nv2.pkl', 'rb') as input:
        timing_nv2 = pickle.load(input)
        [train_time_list_nv2, test_time_list_nv2] = timing_nv2

    train_time_list = []
    test_time_list = []

    for i in range(len(train_time_list_nv1)):
        train_time_list.append(train_time_list_nv1[i] + train_time_list_nv2[i])

    for i in range(len(test_time_list_nv1)):
        test_time_list.append(test_time_list_nv1[i] + test_time_list_nv2[i])

    timing = [train_time_list, test_time_list]

    # Load Results DRNN
    with open('test_results_' + dl_experiments[0] + '_nv1.pkl', 'rb') as input:
        test_DRNN_nv1 = pickle.load(input)
    with open('test_results_' + dl_experiments[0] + '_nv2.pkl', 'rb') as input:
        test_DRNN_nv2 = pickle.load(input)

    test_DRNN = [np.concatenate((test_DRNN_nv1[0], test_DRNN_nv2[0])),
                 np.concatenate((test_DRNN_nv1[1], test_DRNN_nv2[1])),
                 np.concatenate((test_DRNN_nv1[2], test_DRNN_nv2[2]))]

    # Load Results FCN_DAE
    with open('test_results_' + dl_experiments[1] + '_nv1.pkl', 'rb') as input:
        test_FCN_DAE_nv1 = pickle.load(input)
    with open('test_results_' + dl_experiments[1] + '_nv2.pkl', 'rb') as input:
        test_FCN_DAE_nv2 = pickle.load(input)

    test_FCN_DAE = [np.concatenate((test_FCN_DAE_nv1[0], test_FCN_DAE_nv2[0])),
                    np.concatenate((test_FCN_DAE_nv1[1], test_FCN_DAE_nv2[1])),
                    np.concatenate((test_FCN_DAE_nv1[2], test_FCN_DAE_nv2[2]))]

    # Load Results Vanilla L
    with open('test_results_' + dl_experiments[2] + '_nv1.pkl', 'rb') as input:
        test_Vanilla_L_nv1 = pickle.load(input)
    with open('test_results_' + dl_experiments[2] + '_nv2.pkl', 'rb') as input:
        test_Vanilla_L_nv2 = pickle.load(input)

    test_Vanilla_L = [np.concatenate((test_Vanilla_L_nv1[0], test_Vanilla_L_nv2[0])),
                      np.concatenate((test_Vanilla_L_nv1[1], test_Vanilla_L_nv2[1])),
                      np.concatenate((test_Vanilla_L_nv1[2], test_Vanilla_L_nv2[2]))]

    # Load Results Exp Vanilla NL
    with open('test_results_' + dl_experiments[3] + '_nv1.pkl', 'rb') as input:
        test_Vanilla_NL_nv1 = pickle.load(input)
    with open('test_results_' + dl_experiments[3] + '_nv2.pkl', 'rb') as input:
        test_Vanilla_NL_nv2 = pickle.load(input)

    test_Vanilla_NL = [np.concatenate((test_Vanilla_NL_nv1[0], test_Vanilla_NL_nv2[0])),
                       np.concatenate((test_Vanilla_NL_nv1[1], test_Vanilla_NL_nv2[1])),
                       np.concatenate((test_Vanilla_NL_nv1[2], test_Vanilla_NL_nv2[2]))]

    # Load Results Multibranch LANL
    with open('test_results_' + dl_experiments[4] + '_nv1.pkl', 'rb') as input:
        test_Multibranch_LANL_nv1 = pickle.load(input)
    with open('test_results_' + dl_experiments[4] + '_nv2.pkl', 'rb') as input:
        test_Multibranch_LANL_nv2 = pickle.load(input)

    test_Multibranch_LANL = [np.concatenate((test_Multibranch_LANL_nv1[0], test_Multibranch_LANL_nv2[0])),
                             np.concatenate((test_Multibranch_LANL_nv1[1], test_Multibranch_LANL_nv2[1])),
                             np.concatenate((test_Multibranch_LANL_nv1[2], test_Multibranch_LANL_nv2[2]))]

    # Load Results Multibranch LANLD
    with open('test_results_' + dl_experiments[5] + '_nv1.pkl', 'rb') as input:
        test_Multibranch_LANLD_nv1 = pickle.load(input)
    with open('test_results_' + dl_experiments[5] + '_nv2.pkl', 'rb') as input:
        test_Multibranch_LANLD_nv2 = pickle.load(input)

    test_Multibranch_LANLD = [np.concatenate((test_Multibranch_LANLD_nv1[0], test_Multibranch_LANLD_nv2[0])),
                              np.concatenate((test_Multibranch_LANLD_nv1[1], test_Multibranch_LANLD_nv2[1])),
                              np.concatenate((test_Multibranch_LANLD_nv1[2], test_Multibranch_LANLD_nv2[2]))]

    # Load Result FIR Filter
    with open('test_results_FIR_nv1.pkl', 'rb') as input:
        test_FIR_nv1 = pickle.load(input)
    with open('test_results_FIR_nv2.pkl', 'rb') as input:
        test_FIR_nv2 = pickle.load(input)

    test_FIR = [np.concatenate((test_FIR_nv1[0], test_FIR_nv2[0])),
                np.concatenate((test_FIR_nv1[1], test_FIR_nv2[1])),
                np.concatenate((test_FIR_nv1[2], test_FIR_nv2[2]))]

    # Load Result IIR Filter
    with open('test_results_IIR_nv1.pkl', 'rb') as input:
        test_IIR_nv1 = pickle.load(input)
    with open('test_results_IIR_nv2.pkl', 'rb') as input:
        test_IIR_nv2 = pickle.load(input)

    test_IIR = [np.concatenate((test_IIR_nv1[0], test_IIR_nv2[0])),
                np.concatenate((test_IIR_nv1[1], test_IIR_nv2[1])),
                np.concatenate((test_IIR_nv1[2], test_IIR_nv2[2]))]

    ####### Calculate Metrics #######

    print('Calculating metrics ...')

    # DL Metrics

    # Exp FCN-DAE

    [X_test, y_test, y_pred] = test_DRNN

    SSD_values_DL_DRNN = SSD(y_test, y_pred)

    MAD_values_DL_DRNN = MAD(y_test, y_pred)

    PRD_values_DL_DRNN = PRD(y_test, y_pred)

    COS_SIM_values_DL_DRNN = COS_SIM(y_test, y_pred)


    # Exp FCN-DAE

    [X_test, y_test, y_pred] = test_FCN_DAE

    SSD_values_DL_FCN_DAE = SSD(y_test, y_pred)

    MAD_values_DL_FCN_DAE = MAD(y_test, y_pred)

    PRD_values_DL_FCN_DAE = PRD(y_test, y_pred)

    COS_SIM_values_DL_FCN_DAE = COS_SIM(y_test, y_pred)


    # Vanilla L

    [X_test, y_test, y_pred] = test_Vanilla_L

    SSD_values_DL_exp_1 = SSD(y_test, y_pred)

    MAD_values_DL_exp_1 = MAD(y_test, y_pred)

    PRD_values_DL_exp_1 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_1 = COS_SIM(y_test, y_pred)


    # Vanilla_NL

    [X_test, y_test, y_pred] = test_Vanilla_NL

    SSD_values_DL_exp_2 = SSD(y_test, y_pred)

    MAD_values_DL_exp_2 = MAD(y_test, y_pred)

    PRD_values_DL_exp_2 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_2 = COS_SIM(y_test, y_pred)


    # Multibranch_LANL

    [X_test, y_test, y_pred] = test_Multibranch_LANL

    SSD_values_DL_exp_3 = SSD(y_test, y_pred)

    MAD_values_DL_exp_3 = MAD(y_test, y_pred)

    PRD_values_DL_exp_3 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_3 = COS_SIM(y_test, y_pred)


    # Multibranch_LANLD

    [X_test, y_test, y_pred] = test_Multibranch_LANLD

    SSD_values_DL_exp_4 = SSD(y_test, y_pred)

    MAD_values_DL_exp_4 = MAD(y_test, y_pred)

    PRD_values_DL_exp_4 = PRD(y_test, y_pred)

    COS_SIM_values_DL_exp_4 = COS_SIM(y_test, y_pred)


    # Digital Filtering

    # FIR Filtering Metrics
    [X_test, y_test, y_filter] = test_FIR

    SSD_values_FIR = SSD(y_test, y_filter)

    MAD_values_FIR = MAD(y_test, y_filter)

    PRD_values_FIR = PRD(y_test, y_filter)

    COS_SIM_values_FIR = COS_SIM(y_test, y_filter)


    # IIR Filtering Metrics (Best)
    [X_test, y_test, y_filter] = test_IIR

    SSD_values_IIR = SSD(y_test, y_filter)

    MAD_values_IIR = MAD(y_test, y_filter)

    PRD_values_IIR = PRD(y_test, y_filter)

    COS_SIM_values_IIR = COS_SIM(y_test, y_filter)


    ####### Results Visualization #######

    SSD_all = [SSD_values_FIR,
               SSD_values_IIR,
               SSD_values_DL_FCN_DAE,
               SSD_values_DL_DRNN,
               SSD_values_DL_exp_1,
               SSD_values_DL_exp_2,
               SSD_values_DL_exp_3,
               SSD_values_DL_exp_4,
               ]

    MAD_all = [MAD_values_FIR,
               MAD_values_IIR,
               MAD_values_DL_FCN_DAE,
               MAD_values_DL_DRNN,
               MAD_values_DL_exp_1,
               MAD_values_DL_exp_2,
               MAD_values_DL_exp_3,
               MAD_values_DL_exp_4,
               ]

    PRD_all = [PRD_values_FIR,
               PRD_values_IIR,
               PRD_values_DL_FCN_DAE,
               PRD_values_DL_DRNN,
               PRD_values_DL_exp_1,
               PRD_values_DL_exp_2,
               PRD_values_DL_exp_3,
               PRD_values_DL_exp_4,
               ]

    COS_SIM_all = [COS_SIM_values_FIR,
                   COS_SIM_values_IIR,
                   COS_SIM_values_DL_FCN_DAE,
                   COS_SIM_values_DL_DRNN,
                   COS_SIM_values_DL_exp_1,
                   COS_SIM_values_DL_exp_2,
                   COS_SIM_values_DL_exp_3,
                   COS_SIM_values_DL_exp_4,
                   ]


    Exp_names = ['FIR Filter', 'IIR Filter'] + dl_experiments
    
    metrics = ['SSD', 'MAD', 'PRD', 'COS_SIM']
    metric_values = [SSD_all, MAD_all, PRD_all, COS_SIM_all]

    # Metrics table
    vs.generate_table(metrics, metric_values, Exp_names)

    # Timing table
    timing_var = ['training', 'test']
    vs.generate_table_time(timing_var, timing, Exp_names, gpu=True)

    ################################################################################################################
    # Segmentation by noise amplitude
    rnd_test = np.load('rnd_test.npy')

    rnd_test = np.concatenate([rnd_test, rnd_test])

    segm = [0.2, 0.6, 1.0, 1.5, 2.0]  # real number of segmentations is len(segmentations) - 1
    SSD_seg_all = []
    MAD_seg_all = []
    PRD_seg_all = []
    COS_SIM_seg_all = []

    for idx_exp in range(len(Exp_names)):
        SSD_seg = [None] * (len(segm) - 1)
        MAD_seg = [None] * (len(segm) - 1)
        PRD_seg = [None] * (len(segm) - 1)
        COS_SIM_seg = [None] * (len(segm) - 1)
        for idx_seg in range(len(segm) - 1):
            SSD_seg[idx_seg] = []
            MAD_seg[idx_seg] = []
            PRD_seg[idx_seg] = []
            COS_SIM_seg[idx_seg] = []
            for idx in range(len(rnd_test)):
                # Object under analysis (oua)
                # SSD
                oua = SSD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    SSD_seg[idx_seg].append(oua)

                # MAD
                oua = MAD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    MAD_seg[idx_seg].append(oua)

                # PRD
                oua = PRD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    PRD_seg[idx_seg].append(oua)

                # COS SIM
                oua = COS_SIM_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    COS_SIM_seg[idx_seg].append(oua)

        # Processing the last index
        # SSD
        SSD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = SSD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                SSD_seg[-1].append(oua)

        SSD_seg_all.append(SSD_seg)  # [exp][seg][item]

        # MAD
        MAD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = MAD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                MAD_seg[-1].append(oua)

        MAD_seg_all.append(MAD_seg)  # [exp][seg][item]

        # PRD
        PRD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = PRD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                PRD_seg[-1].append(oua)

        PRD_seg_all.append(PRD_seg)  # [exp][seg][item]

        # COS SIM
        COS_SIM_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = COS_SIM_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                COS_SIM_seg[-1].append(oua)

        COS_SIM_seg_all.append(COS_SIM_seg)  # [exp][seg][item]

    # Printing Tables
    seg_table_column_name = []
    for idx_seg in range(len(segm) - 1):
        column_name = str(segm[idx_seg]) + ' < noise < ' + str(segm[idx_seg + 1])
        seg_table_column_name.append(column_name)

    # SSD Table
    SSD_seg_all = np.array(SSD_seg_all)
    SSD_seg_all = np.swapaxes(SSD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the SSD metric')
    vs.generate_table(seg_table_column_name, SSD_seg_all, Exp_names)

    # MAD Table
    MAD_seg_all = np.array(MAD_seg_all)
    MAD_seg_all = np.swapaxes(MAD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the MAD metric')
    vs.generate_table(seg_table_column_name, MAD_seg_all, Exp_names)

    # PRD Table
    PRD_seg_all = np.array(PRD_seg_all)
    PRD_seg_all = np.swapaxes(PRD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the PRD metric')
    vs.generate_table(seg_table_column_name, PRD_seg_all, Exp_names)

    # COS SIM Table
    COS_SIM_seg_all = np.array(COS_SIM_seg_all)
    COS_SIM_seg_all = np.swapaxes(COS_SIM_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the COS SIM metric')
    vs.generate_table(seg_table_column_name, COS_SIM_seg_all, Exp_names)

    ##############################################################################################################
    # Metrics graphs
    vs.generate_hboxplot(SSD_all, Exp_names, 'SSD (au)', log=False, set_x_axis_size=(0, 100.1))
    vs.generate_hboxplot(MAD_all, Exp_names, 'MAD (au)', log=False, set_x_axis_size=(0, 3.01))
    vs.generate_hboxplot(PRD_all, Exp_names, 'PRD (au)', log=False, set_x_axis_size=(0, 100.1))
    vs.generate_hboxplot(COS_SIM_all, Exp_names, 'Cosine Similarity (0-1)', log=False, set_x_axis_size=(0, 1))

    ################################################################################################################
    # Visualize signals

    signals_index = np.array([110, 210, 410, 810, 1610, 3210, 6410, 12810]) + 10

    ecg_signals2plot = []
    ecgbl_signals2plot = []
    dl_signals2plot = []
    fil_signals2plot = []

    signal_amount = 10

    [X_test, y_test, y_pred] = test_Multibranch_LANLD
    for id in signals_index:
        ecgbl_signals2plot.append(X_test[id])
        ecg_signals2plot.append(y_test[id])
        dl_signals2plot.append(y_pred[id])

    [X_test, y_test, y_filter] = test_IIR
    for id in signals_index:
        fil_signals2plot.append(y_filter[id])

    for i in range(len(signals_index)):
        vs.ecg_view(ecg=ecg_signals2plot[i],
                    ecg_blw=ecgbl_signals2plot[i],
                    ecg_dl=dl_signals2plot[i],
                    ecg_f=fil_signals2plot[i],
                    signal_name=None,
                    beat_no=None)

        vs.ecg_view_diff(ecg=ecg_signals2plot[i],
                         ecg_blw=ecgbl_signals2plot[i],
                         ecg_dl=dl_signals2plot[i],
                         ecg_f=fil_signals2plot[i],
                         signal_name=None,
                         beat_no=None)





