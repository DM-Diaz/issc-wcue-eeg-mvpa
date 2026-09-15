rootDir = 'C:\0_Projects\ISSC_wCue_E3\'
rawDir    = [rootDir 'raw' filesep];
preICADir = [rootDir 'preICA' filesep];
ppDir     = [rootDir 'pp' filesep];


% initial preprocessing
SampleRate = 250;
HPfilter = 0.2;
LPfilter = 20;
fileList = dir([rawDir '*.eeg']);

REf_path='C:\matlabToolbox\eeglab2025.0.0\plugins\dipfit\standard_BEM\elec\';
for S =1
    NrChannels = 63; % if no EOG, then 63; with EOG, then 64
    EEG = pop_loadbv(rawDir, ['S' num2str(S) '.vhdr'], [], [1:NrChannels]);
    EEG = pop_resample(EEG, SampleRate);
    EEG=  pop_chanedit(EEG, 'append',63,'changefield',{64,'labels','Cz'},'lookup',[REf_path 'standard_1005.elc'],'setref',{'1:62','Cz'});
    EEG = pop_eegfiltnew(EEG, 'locutoff', HPfilter,'plotfreqz',1);
    EEG = pop_eegfiltnew(EEG, 'hicutoff', LPfilter,'plotfreqz',1);
    EEG = pop_erplabDeleteTimeSegments( EEG , 'afterEventcodeBufferMS',  1000, 'beforeEventcodeBufferMS',  200, 'displayEEG',  1, 'ignoreBoundary',  1, 'ignoreUseType', 'ignore', 'timeThresholdMS',  2000 );
    EEG.setname = 'preICA';
    EEG = pop_saveset( EEG, 'filename', ['S' num2str(S) '_' EEG.setname '.set'] ,'filepath',preICADir);



    % try ICA first time, downsample to speed up ICA, perhaps best to further filter to 2hz
    EEG = pop_resample(EEG, 100); % faster process
    EEG = pop_eegfiltnew(EEG, 'locutoff', 2,'plotfreqz',1);
    EEG = pop_runica(EEG, 'icatype', 'runica','extended',1,'rndreset','yes');
    EEG = pop_iclabel(EEG, 'default');
    EEG.setname = 'ICAv1';
    EEG = pop_saveset( EEG, 'filename', ['S' num2str(S) '_' EEG.setname '.set'] ,'filepath',ppDir);

    close all;clear EEG;
end


