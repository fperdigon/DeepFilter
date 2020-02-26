%% Noise MIT BIH stress Preprocessing

% This sctrip performs a preprocess stage on the MIT BIH stress database.

clear; close all; clc;

%% Installing the Physionet Toolbox

addpath wfdb-app-toolbox-0-10-0/mcode/
addpath qt-database-1.0.0

%% Reading the noise

Path = '/mit-bih-noise-stress-test-database-1.0.0/bw.dat';
[signal,Fs,tm]=rdsamp(Path);

NoiseBWL.channel1 = signal(:,1);
NoiseBWL.channel2 = signal(:,2);

save NoiseBWL NoiseBWL;

