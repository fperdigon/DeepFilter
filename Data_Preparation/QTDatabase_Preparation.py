# -*- coding: utf-8 -*-
# ============================================================
#
#  QTDatabase Preparation 
#  
#
#
# the basis for training purposes. The QTDatabase.mat is the result of this script. 
# This result is organized in a cell array in which
# each ECG signal contains its corresponding beats separated beat by beat.
# It was selected the 1st channel of the ECG signals and the sample rate
# was changed to 360Hz.


#  Before running this section, download the QTdatabase, 
#  and install the Physionet WFDB package
#
#  QTdatabase: https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip
#  Installing Physionet WFDB package run from your terminal:
#    $ pip install wfdb 
#
# ============================================================
#
#  authors: David Castro Piñol, Francisco Perdigon Romero
#  email: davidpinyol91@gmail.com, fperdigon88@gmail.com
#  github id: Dacapi91, fperdigon
#
# ============================================================

import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.signal import resample_poly
import wfdb
import math

# Desired sampling frecuency
newFs = 360

# Preprocessing signals
QTpath = 'qt-database-1.0.0/'
namesPath = glob.glob("qt-database-1.0.0/*.dat")

# final list that will contain all signals and beats processed
QTDatabaseSignals = list()

for i in namesPath:
    
    # reading signals 
    aux = i.split('.dat')
    signal,fields = wfdb.rdsamp(aux[0])
    qu = len(signal)
        
    # reading annotations
    ann = wfdb.rdann(aux[0],'pu1')
    anntype = ann.symbol
    annSamples = ann.sample
    
    # Obtaining P wave start positions
    Anntype = np.array(anntype)    
    idx = Anntype =='p'
    Pidx = annSamples[idx]
    idxS = Anntype =='('
    Sidx = annSamples[idxS]
    idxR = Anntype == 'N'
    Ridx = annSamples[idxR]
    
    ind = np.zeros(len(Pidx))
    
    for j in range(len(Pidx)): 
        arr = np.where(Pidx[j] > Sidx)
        arr = arr[0]
        ind[j] = arr[-1] 
    
    ind = ind.astype(np.int64)
    Pstart = Sidx[ind]
    
    # Shift 40ms before P wave start
    Pstart = Pstart - int(0.04*fields['fs'])
        
    # Extract first channel
    auxSig = signal[0:qu,0]
    
    # Beats separation and removing outliers
    # Beats separation and removal of the vectors that contain more or equal than
    # two beats based on QRS annotations
    beats = list()
    for k in range(len(Pstart)-1):
        remove = (Ridx>Pstart[k]) & (Ridx<Pstart[k+1])
        if np.sum(remove)<2:
            beats.append(auxSig[Pstart[k]:Pstart[k+1]])
    
    # plt.plot(beats[0])      
   
    # Creating the list that will contain each beat per signal
    beatsRe = list()
      
    # processing each beat
    for k in range(len(beats)):
        # Padding data to avoid edge effects caused by resample
        L = math.ceil(len(beats[k])*newFs/fields['fs'])
        normBeat = list()
        normBeat = list(reversed(beats[k])) + list(beats[k]) + list(reversed(beats[k]))
                                  
        # resample beat by beat and saving it
        res = resample_poly(normBeat,newFs,fields['fs'])
        res = res[L-1:2*L-1]
        beatsRe.append(res)

    # storing all beats in each corresponding signal, list of list
    QTDatabaseSignals.append(beatsRe)
    
np.save('QTDatabase',QTDatabaseSignals)
print('=========================================================')
print('Sucessful processed MIT QT database saved as npy')
        

     




