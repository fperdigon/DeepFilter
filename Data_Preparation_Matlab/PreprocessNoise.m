%% Noise MIT BIH stress Preprocessing

% This sctrip performs a preprocess stage on the MIT BIH stress database.

clear; close all; clc;

%% Installing the Physionet Toolbox

% Before running this section, download the QTdatabase, Physionet
% toolbox and the Noise Stress database and add it to the current folder

% Laguna, P., Mark, R. G., Goldberg, A., & Moody, G. B. (1997, September). 
% A database for evaluation of algorithms for measurement of QT and other waveform intervals 
% in the ECG. In Computers in cardiology 1997 (pp. 673-676). IEEE.

% Laguna P, Mark RG, Goldberger AL, Moody GB. A Database for Evaluation of Algorithms 
% for Measurement of QT and Other Waveform Intervals in the ECG. Computers in Cardiology 24:673-676 (1997).

% QTdatabase: https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip
% Physionet toolbox: https://physionet.org/physiotools/matlab/wfdb-app-matlab/wfdb-app-toolbox-0-10-0.zip
% MIT-BIH Noise Stress Test Database: https://physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip

addpath wfdb-app-toolbox-0-10-0/mcode/
addpath qt-database-1.0.0

%% Reading the noise

Path = '/mit-bih-noise-stress-test-database-1.0.0/bw.dat';
[signal,Fs,tm]=rdsamp(Path);

NoiseBWL.channel1 = signal(:,1);
NoiseBWL.channel2 = signal(:,2);

save NoiseBWL NoiseBWL;

