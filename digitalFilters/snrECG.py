# -*- coding: utf-8 -*-
# ============================================================
#
#  SNR of ECG signals before and after filtering computation
#
#  authors: David Castro Piñol, Francisco Perdigon Romero
#  email: davidpinyol91@gmail.com, fperdigon88@gmail.com
#  github id: Dacapi91, fperdigon
#
# ============================================================

import numpy as np

def snrECGbefore(signal,AddedNoise):
    
    #    signal :     the ECG original free noise signal (must be a list)
    #    AddedNoise:  the noise added to the ECG origianl signal 
    #    snr:         the SNR computed in db
    
    # normalize signal and noise, zero mean
    signal = signal - np.mean(signal)
    AddedNoise = AddedNoise - np.mean(AddedNoise)
    
    # computing the power and sum up all
    Ps = np.sum(signal**2)
    Pn = np.sum(AddedNoise**2)
    
    # computing the SNR value
    snr = 10*np.log10(Ps/Pn)
        
    return snr

def snrECGafter(signalF,signal):
    
    #    signal :     the ECG original free noise signal (must be a list)
    #    signalF:     ECG signal filtered.
    #    snr:         the SNR computed in db
    
    # obtaining noise
    noise = signalF - signal
    
    # normalize signal and noise
    signal = signal - np.mean(signal)
    noise = noise - np.mean(noise)
    
    # computing the power and sum up all
    Ps = np.sum(signal**2)
    Pn = np.sum(noise**2)
    
    # computing the SNR value
    snr = 10*np.log10(Ps/Pn)  
    
    return snr

