# -*- coding: utf-8 -*-
# ============================================================
#
#  BWL FIR Filtering 

#  author: David Castro Piñol
#  email: davidpinyol91@gmail.com
#  github id: Dacapi91
#
# ============================================================

import scipy.io as sio
import matplotlib.pyplot as plt
from numpy import log10, ones
from scipy.signal import kaiserord, firwin, filtfilt


def FIRRemoveBL(ecgy,Fs,Fc):
    
#    ecgy:        the contamined signal (must be a list)
#    Fc:          cut-off frequency
#    Fs:          sample frequiency
#    ECG_Clean :  processed signal without BLW
    
    # getting the length of the signal
    signal_len = len(ecgy)
    
    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2.0
    
    # The desired width of the transition from stop to pass,
    # relative to the Nyquist rate. 
    width = 0.07/nyq_rate 
    
    # Attenuation in the stop band, in dB.
    ripple_db = round(-20*log10(0.001))+1 # related to devs in Matlab 
    
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
       
    # Use firwin with a Kaiser window to create a highpass FIR filter.
    h = firwin(N, Fc/nyq_rate, window=('kaiser', beta), pass_zero = False)
    
    # Check filtfilt condition
    if N*3>signal_len:
        diff = N*3- signal_len
        ecgy.extend(list(reversed(ecgy)))   
        ecgy.extend(list(ecgy[-1]*ones(diff)))    
    
    # Filtering with filtfilt
    ECG_Clean = filtfilt(h, 1.0, ecgy)
    ECG_Clean = ECG_Clean[0:signal_len] 
        
    return ECG_Clean,N

# signal for demonstration.
ecgy = sio.loadmat('ecgbeat.mat')
signal = ecgy['ecgy']
signal = list(signal[:,0])

## parameters
Fs = 360
Fc = 0.67

ECG_Clean,N = FIRRemoveBL(signal,Fs,Fc)

plt.figure()
plt.plot(signal[0:len(ecgy['ecgy'])])
plt.plot(ECG_Clean)
plt.show()
plt.figure()











