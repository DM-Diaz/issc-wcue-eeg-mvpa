rootDir   = '/home/moon/Desktop/Dylan_Scripts_Summer_ISSC_wCue/EEG_Matlab/Data/All_Participants';
rawDir    = [rootDir 'Raw' filesep];
preICADir = [rootDir 'preICA_test' filesep];
ppDir     = [rootDir 'pp_test' filesep];

% preprocessing parameters
SampleRate = 250;
HPfilter   = 0.2;
LPfilter   = 20;
NrChannels = 63;   % 63 if no EOG channel, 64 if an EOG channel is present (if EOG is chan. # 64, setting to 63 will exclude it automatically)
REf_path   = 'plugins/dipfit/standard_BEM/elec/';

% recordings to skip: use the exact base filename, no extension. {} = process all
excludeList = {};
% low behavioral performance: 'S19', 'S30', 'S32', 'S39', 'S44', 'S62', 'S63'

% drive the loop from the raw .vhdr files present on disk
rawFiles = dir([rawDir '*.vhdr']);
if isempty(rawFiles)
    error('No .vhdr files found in %s', rawDir);
end

for f = 1:numel(rawFiles)
    vhdrName = rawFiles(f).name;            % 'S60.vhdr' or 'S60 Part 2.vhdr'
    [~, baseName] = fileparts(vhdrName);    % 'S60'      or 'S60 Part 2'

    % skip excluded recordings
    if ismember(baseName, excludeList)
        fprintf('Skipping (excluded): %s\n', baseName);
        continue;
    end

    % skip if already processed, so you can resume an interrupted batch
    % (delete this block if you want re-runs to overwrite)
    if isfile([ppDir baseName '_ICAv1.set'])
        fprintf('Skipping (already done): %s\n', baseName);
        continue;
    end

    fprintf('\n==== Processing %s ====\n', baseName);

    try
        EEG = pop_loadbv(rawDir, vhdrName, [], 1:NrChannels);
        EEG = pop_resample(EEG, SampleRate);
        EEG = pop_chanedit(EEG, 'append',63, 'changefield',{64,'labels','Cz'}, ...
                           'lookup',[REf_path 'standard_1005.elc'], 'setref',{'1:62','Cz'});
        EEG = pop_eegfiltnew(EEG, 'locutoff', HPfilter, 'plotfreqz',0);
        EEG = pop_eegfiltnew(EEG, 'hicutoff', LPfilter, 'plotfreqz',0);
        EEG = pop_erplabDeleteTimeSegments(EEG, 'afterEventcodeBufferMS',1000, ...
                  'beforeEventcodeBufferMS',200, 'displayEEG',0, 'ignoreBoundary',1, ...
                  'ignoreUseType','ignore', 'timeThresholdMS',4000);
        EEG.setname = 'preICA';
        EEG = pop_saveset(EEG, 'filename',[baseName '_preICA.set'], 'filepath',preICADir);

        % first ICA pass (downsample + 2 Hz HP for a cleaner, faster decomposition)
        EEG = pop_resample(EEG, 100);
        EEG = pop_eegfiltnew(EEG, 'locutoff', 2, 'plotfreqz',0);
        EEG = pop_runica(EEG, 'icatype','runica', 'extended',1, 'rndreset','yes');
        EEG = pop_iclabel(EEG, 'default');
        EEG.setname = 'ICAv1';
        EEG = pop_saveset(EEG, 'filename',[baseName '_ICAv1.set'], 'filepath',ppDir);

    catch ME
        fprintf(2, 'ERROR on %s: %s  (skipping)\n', baseName, ME.message);
    end

    close all; clear EEG;
end

fprintf('\nBatch complete.\n');