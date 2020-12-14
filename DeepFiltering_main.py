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
import tensorflow as tf

from utils.metrics import MAD, SSD, PRD, COS_SIM
from utils import visualization as vs
from utils import data_preparation as dp

from digitalFilters.dfilters import FIR_test_Dataset, IIR_test_Dataset
from deepFilter.dl_pipeline import train_dl, test_dl



if __name__ == "__main__":

    dl_experiments = ['DRNN',
                      'FCN-DAE',
                      'Vanilla L',
                      'Vanilla NL',
                      'Multibranch LANL',
                      'Multibranch LANLD'
                      ]

    Dataset = dp.Data_Preparation()

    # Save dataset
    with open('data/dataset.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(Dataset, output)
    print('Dataset saved')

    # Load dataset
    with open('data/dataset.pkl', 'rb') as input:
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
        with open('test_results_' + dl_experiments[experiment] +'.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(test_results, output)
        print('Results from experiment ' + dl_experiments[experiment] + ' saved')

    # Classical Filters

    # FIR
    start_test = datetime.now()
    [X_test_f, y_test_f, y_filter] = FIR_test_Dataset(Dataset)
    end_test = datetime.now()
    train_time_list.append(0)
    test_time_list.append(end_test - start_test)

    test_results_FIR = [X_test_f, y_test_f, y_filter]

    # Save FIR filter results
    with open('test_results_FIR.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(test_results_FIR, output)
    print('Results from experiment FIR filter saved')

    # IIR
    start_test = datetime.now()
    [X_test_f, y_test_f, y_filter] = IIR_test_Dataset(Dataset)
    end_test = datetime.now()
    train_time_list.append(0)
    test_time_list.append(end_test - start_test)

    test_results_IIR = [X_test_f, y_test_f, y_filter]

    # Save IIR filter results
    with open('test_results_IIR.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(test_results_IIR, output)
    print('Results from experiment IIR filter saved')

    # Saving timing list
    timing = [train_time_list, test_time_list]
    with open('timing.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(timing, output)
    print('Timing saved')


    ####### LOAD EXPERIMENTS #######

    # Load timing
    with open('timing.pkl', 'rb') as input:
        timing = pickle.load(input)
        [train_time_list, test_time_list] = timing

    # Load Results DRNN
    with open('test_results_' + dl_experiments[0] + '.pkl', 'rb') as input:
       test_DRNN = pickle.load(input)

    # Load Results FCN_DAE
    with open('test_results_' + dl_experiments[1] +'.pkl', 'rb') as input:
        test_FCN_DAE = pickle.load(input)

    # Load Results Vanilla L
    with open('test_results_' + dl_experiments[2] +'.pkl', 'rb') as input:
        test_Vanilla_L = pickle.load(input)

    # Load Results Exp Vanilla NL
    with open('test_results_' + dl_experiments[3] +'.pkl', 'rb') as input:
        test_Vanilla_NL = pickle.load(input)

    # Load Results Multibranch LANL
    with open('test_results_' + dl_experiments[4] +'.pkl', 'rb') as input:
        test_Multibranch_LANL = pickle.load(input)

    # Load Results Multibranch LANLD
    with open('test_results_' + dl_experiments[5] +'.pkl', 'rb') as input:
        test_Multibranch_LANLD = pickle.load(input)

    # Load Result FIR Filter
    with open('test_results_FIR.pkl', 'rb') as input:
        test_FIR = pickle.load(input)

    # Load Result IIR Filter
    with open('test_results_IIR.pkl', 'rb') as input:
        test_IIR = pickle.load(input)

    ####### Calculate Metrics #######

    ecg_signals2plot = []
    ecgbl_signals2plot = []
    dl_signals2plot = []
    fil_signals2plot = []


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



    for id in signals_id:
        ecgbl_signals2plot.append(X_test[id])
        ecg_signals2plot.append(y_test[id])
        dl_signals2plot.append(y_pred[id])

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

    for id in signals_id:
        fil_signals2plot.append(y_filter[id])

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


    CORR_all = [COS_SIM_values_FIR,
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
    metric_values = [SSD_all, MAD_all, PRD_all, CORR_all]

    # Metrics table
    vs.generate_table(metrics, metric_values, Exp_names)

    # Timing table
    timing_var = ['training', 'test']
    vs.generate_table_time(timing_var, timing, Exp_names)

    vs.generate_hboxplot(SSD_all, Exp_names, 'SSD (au)', log=False, set_x_axis_size=(0, 100.1))
    vs.generate_hboxplot(MAD_all, Exp_names, 'MAD (au)', log=False, set_x_axis_size=(0, 3.01))
    vs.generate_hboxplot(PRD_all, Exp_names, 'PRD (au)', log=False)
    vs.generate_hboxplot(CORR_all, Exp_names, 'Cosine Similarity (0-1)', log=False, set_x_axis_size=(0, 1))

    # Visualize signals
    for i in range(50):  # Plot the 50 first beats from signal sel123
        vs.ecg_view_2(ecg=ecg_signals2plot[i],
                    ecg_blw=ecgbl_signals2plot[i],
                    ecg_dl=dl_signals2plot[i],
                    signal_name=None,
                    beat_no=None)

        vs.ecg_view_diff(ecg=ecg_signals2plot[i],
                         ecg_blw=ecgbl_signals2plot[i],
                         ecg_dl=dl_signals2plot[i],
                         ecg_f=fil_signals2plot[i],
                         signal_name=None,
                         beat_no=None)





