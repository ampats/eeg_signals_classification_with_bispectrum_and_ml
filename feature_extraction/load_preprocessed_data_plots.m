% loads data for all ear-EEG channels, for 1 subject (subNum) and for 1 task (sesNum),
% session number (sesNum) either 2,3,4 or 5 corresponding to "standing", slow walking",
% "fast walking" and "running", respectively

clear all;
startup_bbci_toolbox

%% Load Isolated data
BTB.DataDir = 'path\to\dataset\preprocessed data';

BTB.task = 'ERP'; %ERP, SSVEP
datatype = 'eeg';
%%
switch(BTB.task)
    case 'SSVEP'
        disp_ival= [0 5000]; % SSVEP
        trig_sti = {11,12,13; '5.45','8.57','12'};
        nSub = 23;  %23 kanonika
    case 'ERP'
        disp_ival= [-200 800]; % ERP
        ref_ival= [-200 0] ;
        trig_sti = {2,1 ;'target','non-target'};
        nSub = 1 ;  %24 kanonika
end
%%
CNT = []; 
MRK = [];
EPO = [];
MNT = [];
dataNorm = zeros(512,14) ;

for subNum = 1:nSub
    fprintf('Load Subject %02d ...\n',subNum)

    for sesNum = 2:5
        sub_dire = sprintf('sub-%02d/ses-%02d',subNum,sesNum);
        % sub-01_task-ERP_speed-0.8_scalp-EEG
        naming = sprintf('sub-%02d_ses-%02d_task-%s_%s',...
            subNum,sesNum,BTB.task,datatype);
        filename = fullfile(BTB.DataDir,sub_dire,datatype,naming);

        % load data
        try
            [CNT{subNum,sesNum}, mrk_orig, hdr] = file_readBV(filename, 'Fs', 100);
        catch
            continue;
        end

        % create mrk
        MRK{subNum,sesNum}= mrk_defineClasses(mrk_orig, trig_sti);

        % segmentation
        EPO{subNum,sesNum} = proc_segmentation(CNT{subNum,sesNum}, MRK{subNum,sesNum}, disp_ival);

        MNT= mnt_setElectrodePositions(CNT{subNum,sesNum}.clab);
    end
end

%change these 2 values depending on what subject and task you want to plot for
subNum = 1 ; %subject 1
sesNum = 2 ; %session 2 (standing)

chan = CNT{subNum,sesNum}.clab ;
fs = 100 ;

data = CNT{subNum,sesNum}.x ; % eeg data matrix
for i=33:46 %ear-EEG channels 
    
    % starting data point for each sample, each point corresponds to the starting point of a trial 
    % s1ses2, s1ses3, s1ses5, s2ses2, s2ses3,  s2ses5, ...
    % samplNum = [1034 6139 6955 12035 6168 1859 2762 33124 4252 2881 6165 9605 948 6155 5283 10478 6154 4096 14900 6082 32237 12138 6018 5144 12083 6017 38932 31489 6159 14989 8252 6005 4970 6147 14139 17600 12034 6060 5242 14071 14138 10032 5987 14126 9897] ; 
    % eeg signal standardization instead
    
    % 1034th bin is the starting point for sample of class "standing" for subject 1
    % (change values according to samplNum array, depending on what subject and task you want to plot for)
    % here each channel signal has a length of 512 (5.12 s)
    s = std(data(1034:1545,i)) ;
    dataNorm(:,i-32) = data(6955:7466,i) - mean(data(1034:1545,i)) ;
    dataNorm(:,i-32) = dataNorm(:,i-32) / s ;
    L = length(data(1034:1545,i)) ;

end
