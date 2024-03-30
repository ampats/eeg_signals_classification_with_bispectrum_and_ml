% loads data for all ear-EEG channels, for all subjects (subNum) and for 3 tasks (sesNum),
% sesNum either 2,3,4 or 5 corresponding to "standing", slow walking",
% "fast walking" and "running", respectively
% **all signal duration** (not only 512-point signals)

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
        nSub = 15 ;  %24 kanonika
end
%%
CNT = []; 
MRK = [];
EPO = [];
MNT = [];
data = zeros(91233,nSub*3,14) ; %for all ear-EEG channels (14) and all subjects (15), for the 3 classes
dataNorm = zeros(55000,nSub*3,14) ;
data_stand = zeros(nSub*14, 55000) ;
data_slowWalk = zeros(nSub*14, 55000) ;
data_run = zeros(nSub*14, 55000) ;

i=1 ;
for subNum = 1:nSub
    fprintf('Load Subject %02d ...\n',subNum)

    for sesNum = 2:5

        if sesNum == 4  %skip loading "fast walking" data
            continue;
        end

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

        A = CNT{subNum,sesNum}.x ; % eeg data matrix
        data(1:size(A,1),i,:) = A(:,33:46) ; 
        i = i + 1 ;

    end
end

chan = CNT{subNum,sesNum}.clab ;
fs = 100 ;

%starting data point for each sample, each point corresponds to the starting point of a trial 
%s1ses2, s1ses3, s1ses5, s2ses2, s2ses3,  s2ses5, ...
samplNum = [1034 6139 6955 12035 6168 1859 2762 33124 4252 2881 6165 9605 948 6155 5283 10478 6154 4096 14900 6082 32237 12138 6018 5144 12083 6017 38932 31489 6159 14989 8252 6005 4970 6147 14139 17600 12034 6060 5242 14071 14138 10032 5987 14126 9897] ; 

%eeg signal standardization
for i=1:nSub*3
    for chanNum=1:14

        s = std(data(1:55000,i,chanNum)) ;
        dataNorm(:,i,chanNum) = data(1:55000,i,chanNum) - mean(data(1:55000,i,chanNum)) ;
        dataNorm(:,i,chanNum) = dataNorm(1:55000,i,chanNum) / s ;
        L = length(data(:,i,chanNum)) ;
    end
end
dataNorm = reshape(dataNorm, [3*nSub,14,55000]) ;
pr = reshape(dataNorm(1,:,:), [14,55000]) ;
dataNorm_reshaped = reshape(dataNorm, [3*nSub*14,55000]) ;
csvwrite('data3ALL_wholeDur_eeg.csv', dataNorm_reshaped) ;

ind=1;
for i=1:42:nSub*3*14
    data_stand(ind:ind+13,:) = dataNorm_reshaped(i:i+13,:) ;
    ind = ind+14 ;
end
csvwrite('data3ALL_wholeDur_eeg_task1.csv', data_stand) ;
ind=1;
for i=15:42:nSub*3*14
    data_slowWalk(ind:ind+13,:) = dataNorm_reshaped(i:i+13,:) ;
    ind = ind+14 ;
end
csvwrite('data3ALL_wholeDur_eeg_task2.csv', data_slowWalk) ;
ind=1;
for i=29:42:nSub*3*14
    data_run(ind:ind+13,:) = dataNorm_reshaped(i:i+13,:) ;
    ind = ind+14 ;
end
csvwrite('data3ALL_wholeDur_eeg_task3.csv', data_run) ;