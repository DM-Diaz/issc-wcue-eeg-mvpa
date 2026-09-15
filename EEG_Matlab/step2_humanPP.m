% ============================================================
 % STEP 2 (humanPP route): two-pass ICA cleaning, single script.
 % Runs AFTER step1_ICAv1.m. Sections are separated by %% so you
 % can run each with Ctrl+Enter. Workflow:
 %   1) Run SETUP, then CHUNK 1. It stops at a return().
 %   2) Inspect each S*_postICA.set in eStudio, fill in eyeIC_v2.
 %   3) Run CHUNK 2.
 % If MATLAB was restarted before chunk 2, re-run SETUP first
 % (bad channels are reloaded from disk, so those survive a restart).
 % ============================================================

% ---------- SETUP (needed by both chunks) ----------
rootDir   = '/home/moon/Desktop/Dylan_Scripts_Summer_ISSC_wCue/EEG_Matlab/Data/All_Participants';
rawDir    = [rootDir 'Raw' filesep];
preICADir = [rootDir 'preICA_test' filesep];
ppDir     = [rootDir 'pp_test' filesep];
% 
% % % [] if EOG was dropped at step1 (the 63-channel path); set to its
% % % index if your preICA still contains the EOG channel.
eogChan = [];

% Blink/saccade ICs per subject, S1..S70 in order. [X] in your notes
% became []; those subjects are skipped via excludeList below.
% VERIFY these numbers against your eyeballing before trusting them.
eyeIC_byNum = { ...
  [2 4],[3 8],[2 8],[1 7],[3 7],[1 10],[1],[1 6],[2],[17 34], ...   % S1-S10
  [1 12],[9],[2 3],[4 10],[1],[1 4],[3],[10 14],[],[1], ...         % S11-S20  (S19 excl)
  [2 5],[2 8],[2 11],[6 33],[],[7],[1 5],[2 5],[2 4],[], ...        % S21-S30  (S30 excl)
  [6],[],[13],[],[4],[1 5],[2 6],[2 18],[],[1], ...                 % S31-S40  (S32,S39 excl)
  [],[],[3],[],[4 7],[],[3 4],[1 17],[6 8],[], ...                  % S41-S50  (S44 excl)
  [14],[3],[3 6],[12],[6 9],[],[4],[8 9],[4],[14], ...              % S51-S60
  [4 10],[],[],[5 19],[11],[2 7],[8 21],[5 10],[7 15],[16] ...      % S61-S70  (S62,S63 excl)
};

eyeIC = containers.Map();
for k = 1:numel(eyeIC_byNum)
    eyeIC(sprintf('S%d', k)) = eyeIC_byNum{k};
end

% Duplicate session for S60, keyed by its real filename base (no number slot).
eyeIC('S19') = [5, 15];   % <-- set to the ICs you flagged for that recording
eyeIC('S30') = [3];
eyeIC('S32') = [6];
eyeIC('S39') = [3, 7];
eyeIC('S44') = [5, 10, 15, 40];
eyeIC('S62') = [9, 16];
eyeIC('S63') = [14];

% Recordings to skip even if their .set files exist (your [X] marks).
excludeList = {};
% 'S19','S30','S32','S39','S44','S62','S63'

% ---------- CHUNK 1: remove eyes -> find bad chans -> 2nd ICA -> postICA ----------
setFiles     = dir([ppDir 'S*_ICAv1.set']);
bad_chan_idx = containers.Map();

for f = 1:numel(setFiles)
    icaName  = setFiles(f).name;                 % e.g. 'S60 Part 2_ICAv1.set'
    baseName = erase(icaName, '_ICAv1.set');     % e.g. 'S60 Part 2'

    if ismember(baseName, excludeList)
        fprintf('Skipping (excluded): %s\n', baseName);  continue;
    end
    if ~isKey(eyeIC, baseName)
        warning('No eyeIC entry for %s, skipping (do not process unlabeled).', baseName);
        continue;
    end

    fprintf('\n==== CHUNK 1: %s ====\n', baseName);

    % pass 1: strip eye ICs from ICAv1, then detect bad channels on the result
    EEG = pop_loadset('filename', icaName, 'filepath', ppDir);
    EEG = pop_subcomp(EEG, eyeIC(baseName), 0);
    [EEG, indelec] = pop_rejchan(EEG, 'elec',1:EEG.nbchan, 'threshold',5, 'norm','on', 'measure','kurt');
    indelec(indelec==1)  = [];   % protect Fp1  (confirm this index for your montage)
    indelec(indelec==31) = [];   % protect Fp2
    bad_chan_idx(baseName) = indelec;
    clear EEG;

    % pass 2: second ICA on the good channels of full-resolution preICA
    EEG  = pop_loadset('filename', [baseName '_preICA.set'], 'filepath', preICADir);
    EEG2 = EEG;
    chanidx = 1:EEG.nbchan;
    chanidx(ismember(chanidx, indelec)) = [];
    if ~isempty(eogChan), chanidx(chanidx==eogChan) = []; end
    EEG2 = pop_resample(EEG2, 100);
    EEG2 = pop_eegfiltnew(EEG2, 'locutoff', 2, 'plotfreqz',0);
    EEG2 = pop_runica(EEG2, 'icatype','runica', 'extended',1, 'rndreset','yes', 'chanind',chanidx);

    % transfer weights onto full-channel preICA, label it, save as postICA.
    % (running ICLabel here means the saved postICA shows Eye labels at inspection)
    EEG = pop_editset(EEG, 'setname','postICA', 'icaweights','EEG2.icaweights', ...
                      'icasphere','EEG2.icasphere', 'icachansind','EEG2.icachansind');
    EEG = pop_iclabel(EEG, 'default');
    EEG.setname = 'postICA';
    EEG = pop_saveset(EEG, 'filename',[baseName '_postICA.set'], 'filepath',ppDir);
    fprintf('  2nd ICA on %d channels, weights transferred and labeled.\n', numel(chanidx));
    close('all'); clear EEG EEG2;
end

save([ppDir 'bad_chan_idx.mat'], 'bad_chan_idx');
fprintf('\nCHUNK 1 complete. Inspect each S*_postICA.set, fill eyeIC_v2, then run CHUNK 2.\n');
return   % STOP here: human inspection of postICA happens before chunk 2

%% ---------- CHUNK 2: remove residual eye ICs -> interpolate bad chans -> ICApruned ----------
% Residual blink/saccade ICs found in postICA. Only list subjects that have
% them; any base name not listed here removes nothing. Add entries as you go.
% eyeIC_v2 = containers.Map();
% eyeIC_v2('S1') = [2];
% eyeIC_v2('S2') = [3 7];
% eyeIC_v2('S3') = [2 6];
% eyeIC_v2('S4') = [1 9];
% eyeIC_v2('S5') = [3 7];
% eyeIC_v2('S6') = [1];
% eyeIC_v2('S8') = [1 5];
% eyeIC_v2('S9') = [2];
% eyeIC_v2('S10') = [15 29];
% 
% eyeIC_v2('S11') = [1 11];
% eyeIC_v2('S12') = [9];
% eyeIC_v2('S13') = [2 3];
% eyeIC_v2('S14') = [1 6];
% eyeIC_v2('S15') = [1 36];
% eyeIC_v2('S16') = [1 4];
% eyeIC_v2('S17') = [3];
% eyeIC_v2('S18') = [8 12];
% eyeIC_v2('S20') = [1];
% 
% eyeIC_v2('S21') = [1 4];
% eyeIC_v2('S22') = [1 8];
% eyeIC_v2('S23') = [2 5];
% eyeIC_v2('S24') = [6];
% eyeIC_v2('S25') = [3];
% eyeIC_v2('S26') = [8];
% eyeIC_v2('S27') = [1 5];
% eyeIC_v2('S28') = [2 5];
% eyeIC_v2('S29') = [1 2];
% 
% eyeIC_v2('S31') = [2];
% eyeIC_v2('S33') = [2 12];
% eyeIC_v2('S35') = [4];
% eyeIC_v2('S36') = [1 4];
% eyeIC_v2('S37') = [1 5];
% eyeIC_v2('S38') = [1 14];
% eyeIC_v2('S40') = [1 22];
% 
% eyeIC_v2('S41') = [3 11];
% eyeIC_v2('S43') = [3];
% eyeIC_v2('S45') = [2 4];
% eyeIC_v2('S46') = [16];
% eyeIC_v2('S47') = [1 2];
% eyeIC_v2('S48') = [1 17];
% eyeIC_v2('S49') = [8 4];
% eyeIC_v2('S50') = [1 2];
% 
% eyeIC_v2('S51') = [9];
% eyeIC_v2('S52') = [2];
% eyeIC_v2('S53') = [2 4];
% eyeIC_v2('S54') = [10];
% eyeIC_v2('S55') = [2 5];
% eyeIC_v2('S56') = [3];
% eyeIC_v2('S57') = [3];
% eyeIC_v2('S58') = [7];
% eyeIC_v2('S59') = [2];
% eyeIC_v2('S60') = [14];
% 
% eyeIC_v2('S61') = [5 9];
% eyeIC_v2('S64') = [5];
% eyeIC_v2('S65') = [8];
% eyeIC_v2('S66') = [1 7];
% eyeIC_v2('S67') = [3 15];
% eyeIC_v2('S68') = [2 8];
% eyeIC_v2('S69') = [7 13];
% eyeIC_v2('S70') = [11];
% eyeIC_v2('S60 part 2') = [];

L = load([ppDir 'bad_chan_idx.mat']);  bad_chan_idx = L.bad_chan_idx;

postFiles = dir([ppDir 'S*_postICA.set']);
for f = 1:numel(postFiles)
    baseName = erase(postFiles(f).name, '_postICA.set');
    if ismember(baseName, excludeList)
        fprintf('Skipping (excluded): %s\n', baseName);  continue;
    end

    fprintf('\n==== CHUNK 2: %s ====\n', baseName);

    EEG = pop_loadset('filename',[baseName '_postICA.set'], 'filepath',ppDir);

    if isKey(eyeIC_v2, baseName), comps = eyeIC_v2(baseName); else, comps = []; end
    EEG = pop_subcomp(EEG, comps, 0);

    % interpolate the channels that were excluded from the ICA
    if isKey(bad_chan_idx, baseName) && ~isempty(bad_chan_idx(baseName))
        EEG = pop_interp(EEG, bad_chan_idx(baseName), 'spherical');
    end

    EEG.setname = 'ICApruned';
    EEG = pop_saveset(EEG, 'filename',[baseName '_ICApruned.set'], 'filepath',ppDir);
    clear EEG;
end
fprintf('\nCHUNK 2 complete. Final datasets: S*_ICApruned.set\n');


