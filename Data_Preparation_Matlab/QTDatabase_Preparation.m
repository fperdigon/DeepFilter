%% QTDatabase Preparation 

% This script performs the preparation of the QT-Dataset in order to set
% the basis for traning purposes. The QTDatabase.mat is the result of this script. 
% This result is organized in a cell array in which
% each ECG signal contains its corresponding beats separated beat by beat.
% It was selected the 1st channel of the ECG signals and the sample rate
% was changed to 360Hz.

% Author: MSc Eng. David Castro Piñol
clear; close all; clc;

%% Installing the Physionet Toolbox

% Before running this section, download the QTdatabase and the Physionet
% toolbox and add it to the current folder

% Laguna, P., Mark, R. G., Goldberg, A., & Moody, G. B. (1997, September). 
% A database for evaluation of algorithms for measurement of QT and other waveform intervals 
% in the ECG. In Computers in cardiology 1997 (pp. 673-676). IEEE.

% QTdatabase: https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip
% Physionet toolbox: https://physionet.org/physiotools/matlab/wfdb-app-matlab/wfdb-app-toolbox-0-10-0.zip

addpath wfdb-app-toolbox-0-10-0/mcode/
addpath qt-database-1.0.0

%% Preprocessing signals

% Preparing the paths of the signals
QTpath = 'qt-database-1.0.0/';
content = dir(QTpath);
names = {content.name} ;
ok = regexpi(names, '.*\.(dat)$', 'start') ;
names = names(~cellfun(@isempty,ok)) ;
namesPath = cell(size(names));

for i = 1:length(names)
  namesPath{i} = fullfile(QTpath,names{i}) ;
end

% Creation of Database sctructure
QTDatabase.signals = cell(length(names),1);
QTDatabase.Fs = 360; % The final Fs required
QTDatabase.Names = names;

% Analysing each signal file
for k = 1:length(namesPath)
    
    % Reading signals and annotations
    [signal,Fs,tm]=rdsamp(namesPath{k});
    qu = length(signal);
    [ann,anntype,subtype,chan,num,comments] = rdann(namesPath{k},'pu1',[],qu);
    
    %% Obtaining P wave start positions
    
    idx = anntype == 'p';
    Pidx = ann(idx);
    idxS = anntype == '(';
    Sidx = ann(idxS);
    idxR = anntype == 'N';
    Ridx = ann(idxR);

    ind = zeros(length(Pidx),1);
    for i=1:length(Pidx)
        ind(i) = find(Pidx(i)> Sidx,1,'last');
    end

    Pstart = Sidx(ind);
    
    %% Shift 40ms before P wave start
    
    Pstart = Pstart - 0.04*Fs;
    auxSig = signal(1:qu,1); % extract first channel

    %% Beats separation and removing outliers

    % Beats separation and removal of the vectors that contain more than
    % two beats based on QRS annotations

    beats = {}; j = 1;
    for i = 1:length(Pstart)-1    
        remove = and(Ridx>Pstart(i),Ridx<Pstart(i+1));    
        if(sum(remove)<2)
            beats{j} = auxSig(Pstart(i):Pstart(i+1));    
            j = j+1;
        end         
    end
    
    %% Changing the sampling frequency

    newFs = 360;
    %Fs = 250;
    
    beatsRe = cell(length(beats),1);
    
    % Processing each beats
    for i = 1:length(beats)
        
        % Padding data to avoid edge effects caused by resample
        L = ceil(length(beats{i})*newFs/Fs);
        normBeat = [flip(beats{i});beats{i};flip(beats{i})];
        
        % Resample beat by beat
        BeatRe = resample(normBeat,newFs,Fs);
        beatsRe{i} = BeatRe(L:2*L);
            
    end
    
    % storing all beats in each corresponding signal
    QTDatabase.signals{k} = beatsRe;
  

end

%% Saving QT Database 

save QTDatabase QTDatabase;
disp(date);
























