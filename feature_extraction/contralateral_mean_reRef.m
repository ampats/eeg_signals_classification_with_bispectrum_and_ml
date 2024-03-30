% re-references data with Contralateral-mean method

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

for subNum = 1:nSub
    fprintf('Load Subject %02d ...\n',subNum)

    for sesNum = 2:5

        if sesNum == 4  %skip loading "fast walking" data
            continue;
        end

        sub_dire = sprintf('sub-%02d\\ses-%02d',subNum,sesNum);
        % sub-01_task-ERP_speed-0.8_scalp-EEG
        filepath = sprintf('%s\\%s\\%s\\', BTB.DataDir, sub_dire, datatype) ;
        naming = sprintf('sub-%02d_ses-%02d_task-%s_%s',...
            subNum,sesNum,BTB.task,datatype);
        filename = fullfile(BTB.DataDir,sub_dire,datatype,naming);
        filename_vhdr = sprintf('%s.vhdr', naming) ;
        
        %% re-reference data
        
        % Load EEG data for a single subject in BrainVision format (.vhdr)
        EEG = pop_loadbv(filepath, filename_vhdr);
        
        left_ear_electrodes = 1:8;  % Electrodes on the left ear
        right_ear_electrodes = 9:14;  % Electrodes on the right ear

        EEG.data(left_ear_electrodes, :) = EEG.data(left_ear_electrodes, :) - mean(EEG.data(left_ear_electrodes, :), 1);
        EEG.data(right_ear_electrodes, :) = EEG.data(right_ear_electrodes, :) - mean(EEG.data(right_ear_electrodes, :), 1);

        % Save re-referenced data in EEGLAB format (.set)ref_sub-%02d_ses-%02d_task-%s_%s
        filename_set = sprintf('%s.set', naming) ;
        EEG = pop_saveset(EEG, 'filename', filename_set, 'filepath', filepath);
        
    end
    
end

