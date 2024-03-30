% loads data for all ear-EEG channels, for all subjects (subNum) and for 3 tasks (sesNum),
% sesNum either 2,3,4 or 5 corresponding to "standing", slow walking",
% "fast walking" and "running", respectively

clear all;
startup_bbci_toolbox

%% Load Isolated data
BTB.DataDir = 'E:\diplomatiki\dataset\preprocessed data';

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
data = zeros(91233,nSub*3,9) ; %for all ear-EEG channels (14) and all subjects (15), for the 3 classes
dataNorm = zeros(512,nSub*3*9) ; % will exclude the bad channels, so 9 remaining
nChan = 9 ;
counter = 0 ;
samplIndex = 1 ;

k=1 ;
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
        data(1:size(A,1),k,:) = A(:,[33,35,37,38,39,42,44,45,46]) ; 
        k = k + 1 ;

    end
end

chan = CNT{subNum,sesNum}.clab ;
fs = 100 ;

%starting data point for each sample, each point corresponds to the starting point of a trial 
%s1ses2, s1ses3, s1ses5, s2ses2, s2ses3,  s2ses5, ...
samplNum = [1034 6139 6955 12035 6168 1859 2762 33124 4252 2881 6165 9605 948 6155 5283 10478 6154 4096 14900 6082 32237 12138 6018 5144 12083 6017 38932 31489 6159 14989 8252 6005 4970 6147 14139 17600 12034 6060 5242 14071 14138 10032 5987 14126 9897] ; 

%reshaping data matrix
data1 = permute(data, [1 3 2]) ;
data2 = reshape(data1, [91233,9*45]) ;
data_reshaped = transpose(data2) ;
test1 = data(:,5,6) ;
test2 = data2(:,42) ;

%eeg signal standardization
for i=1:nSub*3*nChan
    
    counter = counter + 1 ;
    if counter >=1 && counter <=9
        %getting the first 512 data points of each sample to get 5.12s-windows of data
        s = std(data_reshaped(i,samplNum(samplIndex):samplNum(samplIndex)+511)) ;
        dataNorm(:,i) = data_reshaped(i,samplNum(samplIndex):samplNum(samplIndex)+511) - mean(data_reshaped(i,samplNum(samplIndex):samplNum(samplIndex)+511)) ;
        dataNorm(:,i) = dataNorm(:,i) / s ;
    end
    if counter == 9 
        counter = 0 ;
        samplIndex = samplIndex+1 ;
    end
    
end

dataNorm_trans = transpose(dataNorm) ;
csvwrite('data_no_bad_chans.csv', dataNorm_reshaped) ;