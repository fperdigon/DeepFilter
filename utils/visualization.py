#============================================================
#
#  Deep Learning BLW Filtering
#  Data Visualization
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# conda install -c conda-forge prettytable
from prettytable import PrettyTable

def generate_violinplots(np_data, description, ylabel, log):
    # Process the results and store in Panda objects

    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)


    # Set up the matplotlib figure

    f, ax = plt.subplots()

    sns.set(style="whitegrid")

    ax = sns.violinplot(data=pd_df, palette="Set3", bw=.2, cut=1, linewidth=1)

    if log:
        ax.set_yscale("log")
        #ylabel = 'Log10( ' + ylabel + ' )'

    ax.set(xlabel='Models/Methods', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)

    plt.show()

    #plt.savefig(store_folder + 'violinplot_fco' + info + description + '.png')


def generate_barplot(np_data, description, ylabel, log):
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)


    # Set up the matplotlib figure

    f, ax = plt.subplots()

    sns.set(style="whitegrid")

    ax = sns.barplot(data=pd_df)

    if log:
        ax.set_yscale("log")
        #ylabel = 'Log10( ' + ylabel + ' )'

    ax.set(xlabel='Models/Methods', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)

    plt.show()

    #plt.savefig(store_folder + 'violinplot_fco' + info + description + '.png')

def generate_boxplot(np_data, description, ylabel, log):
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)
    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)


    # Set up the matplotlib figure

    f, ax = plt.subplots()

    sns.set(style="whitegrid")

    ax = sns.boxplot(data=pd_df)

    if log:
        ax.set_yscale("log")
        #ylabel = 'Log10( ' + ylabel + ' )'

    ax.set(xlabel='Models/Methods', ylabel=ylabel)
    ax = sns.despine(left=True, bottom=True)



    plt.show()

    #plt.savefig(store_folder + 'violinplot_fco' + info + description + '.png')


def generate_hboxplot(np_data, description, ylabel, log, set_x_axis_size=None):
    # Process the results and store in Panda objects
    col = description
    loss_val_np = np.rot90(np_data)

    pd_df = pd.DataFrame.from_records(loss_val_np, columns=col)


    # Set up the matplotlib figure
    sns.set(style="whitegrid")

    f, ax = plt.subplots(figsize=(15, 6))

    ax = sns.boxplot(data=pd_df, orient="h", width=0.4)

    if log:
        ax.set_xscale("log")
        #ylabel = 'Log10( ' + ylabel + ' )'

    if set_x_axis_size != None:
        ax.set_xlim(set_x_axis_size)


    ax.set(ylabel='Models/Methods', xlabel=ylabel)
    ax = sns.despine(left=True, bottom=True)



    plt.show()

    #plt.savefig(store_folder + 'violinplot_fco' + info + description + '.png')


def ecg_view(ecg, ecg_blw, ecg_dl, ecg_f, signal_name=None, beat_no=None):

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(ecg_blw, 'k', label='ECG + BLW')
    plt.plot(ecg, 'g', label='ECG orig')
    plt.plot(ecg_dl, 'b', label='ECG DL Filtered')
    plt.plot(ecg_f, 'r', label='ECG IIR Filtered')
    plt.grid(True)

    plt.ylabel('au')
    plt.xlabel('samples')

    leg = ax.legend()

    if signal_name != None and beat_no != None:
        plt.title('Signal ' + str(signal_name) + 'beat ' + str(beat_no))
    else:
        plt.title('ECG signal for comparison')

    plt.show()

def ecg_view_2(ecg, ecg_blw, ecg_dl, signal_name=None, beat_no=None):

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.subplot(311)
    plt.title('ECG original signal')
    plt.plot(ecg, 'g', label='ECG orig')
    plt.grid(True)
    plt.ylabel('au')
    plt.xlabel('samples')
    plt.subplots_adjust(hspace=0.5)


    plt.subplot(312)
    plt.title('ECG original + BLW')
    plt.plot(ecg_blw, 'k', label='ECG + BLW')
    plt.grid(True)
    plt.ylabel('au')
    plt.xlabel('samples')
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(313)
    plt.title('Filtered ECG + BLW signal using DeepFilter')
    plt.plot(ecg_dl, 'b', label='ECG DL Filtered')
    plt.plot(ecg - ecg_dl, color='#CB9590', lw=2, label='Difference ECG - DeepFilter')
    plt.grid(True)
    plt.ylabel('au')
    plt.xlabel('samples')
    plt.legend()
    plt.subplots_adjust(hspace=0.5)

    plt.show()


def ecg_view_diff(ecg, ecg_blw, ecg_dl, ecg_f, signal_name=None, beat_no=None):

    fig, ax = plt.subplots(figsize=(16, 9))
    plt.plot(ecg, 'g', label='ECG orig')
    plt.plot(ecg_dl, 'b', label='ECG DL Filtered')
    plt.plot(ecg_f, 'r', label='ECG IIR Filtered')
    plt.plot(ecg - ecg_dl, color='#0099ff', lw=3, label='Difference ECG - DL Filter')
    plt.plot(ecg - ecg_f, color='#cb828d', lw=3, label='Difference ECG - IIR Filter')
    plt.grid(True)

    plt.ylabel('Amplitude (au)')
    plt.xlabel('samples')

    leg = ax.legend()

    if signal_name != None and beat_no != None:
        plt.title('Signal ' + str(signal_name) + 'beat ' + str(beat_no))
    else:
        plt.title('ECG signal for comparison')

    plt.show()


def generate_table(metrics, metric_values, Exp_names):
    # Print tabular results in the console, in a pretty way

    print('\n')

    tb = PrettyTable()
    ind = 0

    for exp_name in Exp_names:

        tb.field_names = ['Method/Model'] + metrics

        tb_row = []
        tb_row.append(exp_name)

        for metric in metric_values:
            m_mean = np.mean(metric[ind])
            m_std = np.std(metric[ind])
            tb_row.append('{:.3f}'.format(m_mean) + ' (' + '{:.3f}'.format(m_std) + ')')

        tb.add_row(tb_row)
        ind += 1

    print(tb)

def generate_table_time(column_names, all_values, Exp_names):
    # Print tabular results in the console, in a pretty way

    # The FIR and IIR are the last on all_values
    # We need circular shift them to the right
    all_values[0] = all_values[0][-2::] + all_values[0][0:-2]
    all_values[1] = all_values[1][-2::] + all_values[1][0:-2]

    print('\n')

    tb = PrettyTable()
    ind = 0

    for exp_name in Exp_names:

        tb.field_names = ['Method/Model'] + [column_names[0] + '(GPU) h:m:s:ms'] + [column_names[1] + '(GPU) h:m:s:ms']

        tb_row = []
        tb_row.append(exp_name)
        tb_row.append(all_values[0][ind])
        tb_row.append(all_values[1][ind])

        tb.add_row(tb_row)

        ind += 1

    print(tb)