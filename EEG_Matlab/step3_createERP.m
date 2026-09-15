% step3_createERP.m  -- blocks 1-5 (cued phase) only
% cd('/home/moon/matlabToolbox/eeglab2025.0.0'); eeglab;

rootDir   = '/home/moon/Desktop/Dylan_Scripts_Summer_ISSC_wCue/EEG_Matlab/Data/All_Participants';
ppDir_orig = [rootDir 'pp_test' filesep];             % source: Sx_ICApruned.set
ppDir      = [rootDir 'preprocessed_test' filesep];   % output: Sx_preprocessed.set
erpDir     = [rootDir 'erp_BP_test' filesep];         % output: Sx_BP.erp
bdfFile    = [rootDir 'binlister_v2.txt'];       % confirm this file exists / is the intended version
if ~exist(ppDir, 'dir'),  mkdir(ppDir);  end
if ~exist(erpDir, 'dir'), mkdir(erpDir); end
cd(erpDir)

mvpa_sbjList =[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,...
    11, 12, 13, 14, 15, 16, 17, 18, 20, ...
    21, 22, 23, 24, 25, 26, 27, 28, 29, ...
    31, 33, 35, 36, 37, 38, 40, ...
    41, 42, 43, 45, 46, 47, 48, 49, ...
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, ...
    65, 66, 68, 69, 70]; % 58 total

sbjList = mvpa_sbjList;

blockCode   = 9;   % beginning of each block
nKeepBlocks = 5;   % keep blocks 1-5, drop 6+

for S = sbjList

    % now that ICA is done and blinks, saccades are removed
    EEG = pop_loadset('filename',['S' num2str(S) '_ICApruned.set'], 'filepath',ppDir_orig);

    % ---- keep only blocks 1-5: crop at the start of block 6 ----
    % Robust to raw 'S  9' strings (cleaning happens later in the eventlist step).

    % Pull every event's code into a plain number list of integers (instead
    % of strings e.g. 'S 9' --> 9
    nEv = numel(EEG.event); codes = nan(1,nEv);
    for i = 1:nEv
        t = EEG.event(i).type;
        if isnumeric(t), codes(i) = double(t);
        else
            d = regexp(char(t), '\d+', 'match', 'once');
            if ~isempty(d), codes(i) = str2double(d); end
        end
    end
    % Retrive all the blockCode = 9 indices where blockIdx(1), blockIdx(2),
    % and so on...
    blockIdx = find(codes == blockCode);
    
    % Make sure that a 6th marker exists otherwise, we won't know where to
    % cut at
    if numel(blockIdx) < nKeepBlocks + 1
        fprintf(2, 'S%d: only %d block markers, cannot locate block %d -> SKIPPING.\n', ...
                S, numel(blockIdx), nKeepBlocks+1);
        clear EEG; continue;
    end
    
    % Reference the start of block 6 using .latency for time and cut at
    % that point. pop_select keeps data up to one sample before block 6
    % begins
    cutLat = round(EEG.event(blockIdx(nKeepBlocks+1)).latency);   % start of block 6
    EEG = pop_select(EEG, 'point', [1 cutLat-1]);
    fprintf('S%d: %d block markers, cropped at sample %d, kept blocks 1-%d (%.1f s)\n', ...
            S, numel(blockIdx), cutLat, nKeepBlocks, EEG.pnts/EEG.srate);
    % -----------------------------------------------------------

    origEEG = EEG;
    origEEG.nbchan

    % find out bad channels
    [EEG,indelec] = pop_rejchan(EEG, 'elec',[1:origEEG.nbchan] ,'threshold',5,'norm','on','measure','kurt');
    ICApruned_bad_chan_idx{S} = indelec;
    EEG.nbchan
    
    % ---- manual bad-channel removal for specific subjects (from the diagnostic) ----
    manualBad = containers.Map('KeyType','double','ValueType','any');
    manualBad(60) = {'FT7'};
    manualBad(1)  = {'P7'};
    manualBad(7)  = {'O2','P8','PO8','Oz'};
    manualBad(11) = {'P4'};
    manualBad(25) = {'PO8','C4'};
    manualBad(33) = {'PO7'};
    manualBad(35) = {'AFz','AF7'};
    manualBad(41) = {'F4'};
    manualBad(48) = {'TP7','P2','F2'};
    manualBad(51) = {'CP6','Fz'};
    manualBad(53) = {'CP2','C2','P2'};
    manualBad(54) = {'C2','AF8','C1'};
    manualBad(58) = {'FC4','FC6','POz'};
    manualBad(69) = {'Fp2','TP10','CPz','P5'};
    manualBad(70) = {'CPz'};
    manualBad(52) = {'PO4','TP8'};
    manualBad(65) = {'P2','O2'};
    manualBad(2)  = {'P7','O1','C1','PO7','FT7','P5'};
    manualBad(46) = {'O2','CP3','POz','C3','CP4','PO8','F3','CP6'};
    manualBad(49) = {'CP2','Oz','Pz','FC4','F1'};

    if isKey(manualBad, S)
        idx = find(ismember({EEG.chanlocs.labels}, manualBad(S)));
        if ~isempty(idx)
            fprintf('S%d: manually removing %s\n', S, strjoin({EEG.chanlocs(idx).labels}, ', '));
            manual_removed{S} = {EEG.chanlocs(idx).labels};   % record for methods
            EEG = pop_select(EEG, 'nochannel', idx);
        end
    end


    % ---- add Cz back as a data channel, then average-reference ----
    % coordinate values originate from original script's refloc struct
    czTemplate = EEG.chanlocs(1);              % inherit file's exact 11 fields
    czTemplate.labels     = 'Cz';
    czTemplate.type       = '';
    czTemplate.theta      = 177.4959;
    czTemplate.radius     = 0.029055;
    czTemplate.X          = -9.167;
    czTemplate.Y          = -0.4009;
    czTemplate.Z          = 100.244;
    czTemplate.sph_theta  = -177.4959;
    czTemplate.sph_phi    = 84.77;
    czTemplate.sph_radius = 100.6631;
    czTemplate.ref        = '';
    
    EEG.chanlocs(EEG.nbchan+1) = czTemplate;
    EEG.data(EEG.nbchan+1,:)   = 0;            % online reference is flat before re-ref
    EEG.nbchan                 = EEG.nbchan + 1;
    EEG = eeg_checkset(EEG);
    
    EEG = pop_reref(EEG, []);                   % common average reference
    if ~strcmp(EEG.ref, 'average')
        error('S%d: re-reference did not set ref to average', S);
    end
    % ---------------------------------------------------------------


    % interpolate bad channels
    EEG = pop_interp(EEG, origEEG.chanlocs, 'spherical');

    if EEG.nbchan==64 && EEG.srate==250      % was ||, which never really enforced srate
    else
        '???'
        return
    end

    % create event list numeric  (on the cropped, blocks-1-5 data)
    EEG  = pop_creabasiceventlist( EEG , 'AlphanumericCleaning', 'on', 'BoundaryNumeric', { -99 }, 'BoundaryString', { 'boundary' }, 'Eventlist', [erpDir 'S' num2str(S) '_eventlist.txt']);

    % assign bins per the descriptor file
    EEG  = pop_binlister( EEG , 'BDF', bdfFile, 'ExportEL', [erpDir 'S' num2str(S) '_eventlist2.txt'], 'IndexEL',  1, 'SendEL2', 'EEG&Text', 'Voutput', 'EEG' );

    erpWindow = [-200.0 800.0]; % in ms
    EEG = pop_epochbin( EEG , erpWindow,  'pre');

    % Flag epochs containing CRAP or do further blink rejections
    threshold_uV = 100;
    windowSize   = 200;  % ms
    windowStep   = 100;  % ms
    EEG   = pop_artmwppth( EEG , 'Channel',  1:size(EEG.chanlocs,2), 'Flag',  1, 'LowPass',  -1, 'Threshold',  threshold_uV, 'Twindow', erpWindow , 'Windowsize',  windowSize, 'Windowstep',  windowStep );

    % ---- create ERPs FIRSTT, while the flags still exist on EEG ----
    % 'Criterion','good' excludes the flagged epochs, so the .erp is unchanged
    % from before this edit.
    ERP = pop_averager( EEG , 'Criterion', 'good', 'DQ_custom_wins', 0, 'DQ_flag', 1, 'DQ_preavg_txt', 0, 'ExcludeBoundary', 'on', 'SEM', 'on' );
    erpname = ['S' num2str(S) '_BP'];
    ERP = pop_savemyerp(ERP, 'erpname', erpname, 'filename', [erpname '.erp'], 'filepath', erpDir, 'Warning', 'off');

    % write the artifact-detection summary BEFORE removing epochs, so the
    % accepted/rejected counts are still reported (rejepoch would zero them out)
    EEG = pop_summary_AR_eeg_detection(EEG, [erpDir 'S' num2str(S) '_artifactDetection.txt']);
    pop_summary_AR_eeg_detection(EEG, [ppDir 'S' num2str(S) '_artifactDetection.txt']);

    % ---- physically remove flagged epochs so the exported .set is clean ----
    % read_epochs_eeglab does not carry ERPLAB artifact flags into MNE-Python, so the
    % decode was loading flagged epochs as if good. Deleting them here makes the
    % saved .set match exactly the trials that went into the ERP average.
    rejIdx = find(EEG.reject.rejmanual);
    fprintf('S%d: removing %d flagged epochs (%d -> %d) before saving .set\n', ...
            S, numel(rejIdx), EEG.trials, EEG.trials - numel(rejIdx));
    if ~isempty(rejIdx)
        EEG = pop_rejepoch(EEG, rejIdx, 0);   % 0 = no confirmation dialog
    end
    % ----------------------------------------------------------------------------

    % save the CLEAN preprocessed dataset (flagged epochs removed) for MNE
    EEG.setname = ['S' num2str(S) '_preprocessed'];
    EEG = pop_saveset( EEG, 'filename', [EEG.setname '.set'] ,'filepath',ppDir);

    close all; clear EEG;
end


