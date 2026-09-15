% step3_createERP_mvpa_fixed_v5.m
% ------------------------------------------------------------------------------
% MVPA variant of step3_createERP.m. Two additions over the original:
%   (1) Uses binlister_mvpa.txt (adds diagnostic bins 7-9) instead of binlister_v2.
%   (2) Exports a per-epoch metadata CSV aligning each SURVIVING epoch to its
%       behavioral-log row, carrying stimId, block, episode, itSRProb, bkSRProb,
%       congruency, RT, accuracy, etc. This CSV is what makes the diagnostic (B),
%       neutral-block (A-neutral), and leave-one-item-out (C) analyses pure Python.
%
% ------------------------------------------------------------------------------
% close all; clearvars;
% cd('/home/moon/matlabToolbox/eeglab2025.0.0'); eeglab;

rootDir   = '/home/moon/Desktop/Dylan_Scripts_Summer_ISSC_wCue/EEG_Matlab/Data/All_Participants';
ppDir_orig = [rootDir 'pp_test' filesep];                 % source: Sx_ICApruned.set
ppDir      = [rootDir 'preprocessed_test' filesep];  % output: clean epoched sets
erpDir     = [rootDir 'erp_BP_test' filesep];        % output: erp + eventlists
metaDir    = [rootDir 'mvpa_results_test' filesep 'metadata' filesep];
bdfFile    = [rootDir 'binlister_mvpa.txt'];
behFile    = [rootDir 'E3_N70.xlsx'];
for d = {ppDir, erpDir, metaDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end
cd(erpDir)

% ---- behavioral log (all subjects) ----
beh = readtable(behFile);

% ---- final keep-list (hard exclusions already removed) ----
sbjList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
           11, 12, 13, 14, 15, 16, 17, 18, 20, ...
           21, 22, 23, 24, 25, 26, 27, 28, 29, ...
           31, 33, 35, 36, 37, 38, 40, ...
           41, 42, 43, 45, 46, 47, 48, 49, ...
           51, 52, 53, 54, 55, 56, 57, 58, 59, 60, ...
           65, 66, 68, 69, 70];

blockCode   = 9;   % beginning of each block
nKeepBlocks = 5;   % keep blocks 1-5 (cued phase)
itemCodes   = [1 2 3];   % item-cue trigger values

% ---- manual bad channels (same as the ERP re-run) ----
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

for S = sbjList
    fprintf('\n==== S%d ====\n', S);
    EEG = pop_loadset('filename',['S' num2str(S) '_ICApruned.set'], 'filepath',ppDir_orig);

    % ---- crop to blocks 1-5 (same logic as the ERP pipeline) ----
    codes = local_codes(EEG);
    blockIdx = find(codes == blockCode);
    if numel(blockIdx) < nKeepBlocks + 1
        fprintf(2,'S%d: only %d block markers, cannot locate block 6 -> SKIP\n', S, numel(blockIdx));
        clear EEG; continue;
    end
    % Crop from the first retained block marker through the sample immediately
    % before block 6. Starting at block 1 also prevents any practice events that
    % precede the first block marker from entering the metadata sequence.
    startLat = round(EEG.event(blockIdx(1)).latency);
    cutLat   = round(EEG.event(blockIdx(nKeepBlocks+1)).latency);
    EEG = pop_select(EEG, 'point', [startLat cutLat-1]);

    % Capture the retained item-cue sequence from the CROPPED continuous data.
    % We use this only to verify the behavioral ordering. We deliberately do NOT
    % use urevent IDs for metadata alignment: ERPLAB may rebuild/reindex those IDs
    % during event-list/bin/epoch operations, so pre- and post-epoch IDs need not
    % belong to the same numbering system.
    croppedCodes = local_codes(EEG);
    croppedItemEventIdx = find(ismember(croppedCodes, itemCodes));
    eeg_item_seq = croppedCodes(croppedItemEventIdx);

    % The BDF does NOT bin every item cue. Its exact pattern is:
    %   previous feedback 11 . item 1/2/3, task 4-7, response 13-16, feedback 11
    % Therefore only current-correct, post-correct trials receive bins. Build the
    % same eligibility mask over the continuous-data trials so the post-epoch data
    % can be aligned to the corresponding behavioral subset.
    binEligible = false(1, numel(croppedItemEventIdx));
    for tr = 1:numel(croppedItemEventIdx)
        p = croppedItemEventIdx(tr);
        if p > 1 && (p + 3) <= numel(croppedCodes)
            binEligible(tr) = ...
                croppedCodes(p-1) == 11 && ...
                ismember(croppedCodes(p),   itemCodes) && ...
                ismember(croppedCodes(p+1), 4:7) && ...
                ismember(croppedCodes(p+2), 13:16) && ...
                croppedCodes(p+3) == 11;
        end
    end
    expectedEpochItemSeq = eeg_item_seq(binEligible);

    % Detect a mid-block recording seam (merged files) BEFORE any further event
    % processing, while EEG.event still holds the continuous stream. [] if none.
    seamHint = local_merge_seam_items(EEG, itemCodes);

    origEEG = EEG;

    % ---- bad-channel detection + manual removal ----
    [EEG, ~] = pop_rejchan(EEG, 'elec',1:origEEG.nbchan, 'threshold',5, 'norm','on', 'measure','kurt');
    if isKey(manualBad, S)
        idx = find(ismember({EEG.chanlocs.labels}, manualBad(S)));
        if ~isempty(idx)
            fprintf('  manually removing %s\n', strjoin({EEG.chanlocs(idx).labels}, ', '));
            EEG = pop_select(EEG, 'nochannel', idx);
        end
    end

    % ---- add Cz back, average reference ----
    czTemplate = EEG.chanlocs(1);
    czTemplate.labels='Cz'; czTemplate.type='';
    czTemplate.theta=177.4959; czTemplate.radius=0.029055;
    czTemplate.X=-9.167; czTemplate.Y=-0.4009; czTemplate.Z=100.244;
    czTemplate.sph_theta=-177.4959; czTemplate.sph_phi=84.77; czTemplate.sph_radius=100.6631;
    czTemplate.ref='';
    EEG.chanlocs(EEG.nbchan+1) = czTemplate;
    EEG.data(EEG.nbchan+1,:)   = 0;
    EEG.nbchan                 = EEG.nbchan + 1;
    EEG = eeg_checkset(EEG);
    EEG = pop_reref(EEG, []);
    if ~strcmp(EEG.ref, 'average'), error('S%d: reref failed', S); end

    % ---- interpolate removed channels back to full montage ----
    EEG = pop_interp(EEG, origEEG.chanlocs, 'spherical');
    if ~(EEG.nbchan==64 && EEG.srate==250), error('S%d: unexpected montage/srate', S); end

    % ---- eventlist + binlister (MVPA bins incl. diagnostic) + epoch ----
    EEG = pop_creabasiceventlist(EEG, 'AlphanumericCleaning','on', ...
              'BoundaryNumeric',{-99}, 'BoundaryString',{'boundary'}, ...
              'Eventlist',[erpDir 'S' num2str(S) '_eventlist.txt']);
    EEG = pop_binlister(EEG, 'BDF',bdfFile, ...
              'ExportEL',[erpDir 'S' num2str(S) '_eventlist2.txt'], ...
              'IndexEL',1, 'SendEL2','EEG&Text', 'Voutput','EEG');
    erpWindow = [-200.0 800.0];
    EEG = pop_epochbin(EEG, erpWindow, 'pre');

    % ---- artifact detection (flag, not yet remove) ----
    EEG = pop_artmwppth(EEG, 'Channel',1:size(EEG.chanlocs,2), 'Flag',1, ...
              'LowPass',-1, 'Threshold',100, 'Twindow',erpWindow, ...
              'Windowsize',200, 'Windowstep',100);

    % =========================================================================
    %  METADATA ALIGNMENT  (do this BEFORE pop_rejepoch)
    %  On any failure we set behRowsPresent = [] and skip ONLY the metadata; the
    %  ERP average and clean .set below are always written.
    % =========================================================================

    % (a) EEG side was captured immediately after cropping (eeg_item_seq).

    % (b) behavioral side: this subject's cued-phase rows in trial order
    bs = beh(beh.sbjId==S & ismember(beh.blockId,1:nKeepBlocks), :);
    bs = sortrows(bs, {'blockId','trialId'});
    beh_item_seq = double(bs.triggerCode1(:)');
    fprintf('  retained item cues: EEG=%d, behavior=%d\n', ...
            numel(eeg_item_seq), numel(beh_item_seq));

    % (c) ALIGN: exact when counts match, gap-aware when the EEG recorded fewer
    %     trials than the task presented. The 1/2/3 fingerprint locates the gap;
    %     the recording seam disambiguates when the fingerprint is ambiguous.
    nE = numel(eeg_item_seq); nB = numel(beh_item_seq);
    behRowsPresent = [];
    if nE == nB && isequal(eeg_item_seq, beh_item_seq)
        behRowsPresent = 1:nB;
        fprintf('  alignment: exact (%d trials)\n', nB);
    elseif nE < nB
        [rowsTmp, gapMsg] = local_gap_align(eeg_item_seq, beh_item_seq, seamHint);
        if isempty(rowsTmp)
            fprintf(2,'  S%d: %s -> metadata SKIPPED\n', S, gapMsg);
        else
            behRowsPresent = rowsTmp;
            fprintf('  alignment: %s\n', gapMsg);
        end
    else
        fprintf(2,'  S%d: EEG has MORE item cues than behavior (%d > %d) -> metadata SKIPPED\n', ...
                S, nE, nB);
    end

    % ---- metadata (only if alignment succeeded) ----
    if ~isempty(behRowsPresent)
        bs = bs(behRowsPresent, :);

        % (d) pop_epochbin creates one epoch per BIN-ELIGIBLE item event. Align to
        % that exact behavioral subset (current-correct and post-correct trials).
        bsEpoch = bs(binEligible, :);
        expectedBehSeq = double(bsEpoch.triggerCode1(:)');
        fprintf('  BDF-eligible trials: %d/%d (current-correct and post-correct)\n', ...
                height(bsEpoch), height(bs));

        nEp = EEG.trials;
        epoch_item_seq = nan(1,nEp);
        for ep = 1:nEp
            epoch_item_seq(ep) = local_epoch_item_code(EEG, ep, itemCodes);
        end

        metaOk = true;
        if ~isequal(expectedEpochItemSeq, expectedBehSeq)
            fprintf(2,'  S%d: internal eligible-sequence mismatch before epoching -> metadata SKIPPED\n', S);
            metaOk = false;
        elseif nEp ~= height(bsEpoch)
            fprintf(2,'  S%d: post-epoch count mismatch EEG=%d BDF-eligible behavior=%d -> metadata SKIPPED\n', ...
                    S, nEp, height(bsEpoch));
            metaOk = false;
        elseif any(isnan(epoch_item_seq))
            bad = find(isnan(epoch_item_seq));
            fprintf(2,'  S%d: could not read the time-zero item code for %d epochs (first: %s) -> metadata SKIPPED\n', ...
                    S, numel(bad), mat2str(bad(1:min(10,numel(bad)))));
            metaOk = false;
        elseif ~isequal(epoch_item_seq, expectedEpochItemSeq)
            bad = find(epoch_item_seq ~= expectedEpochItemSeq);
            fprintf(2,'  S%d: post-epoch item sequence mismatch (%d/%d differ) -> metadata SKIPPED\n', ...
                    S, numel(bad), nEp);
            show = bad(1:min(10,numel(bad)));
            fprintf(2,'       first mismatches [epoch EEG expected]:\n');
            fprintf(2,'       %d  %d  %d\n', ...
                    [show(:), epoch_item_seq(show(:))', expectedEpochItemSeq(show(:))']');
            metaOk = false;
        end

        if metaOk
            fprintf('  post-epoch sequence verified: %d/%d BDF-eligible epochs aligned\n', ...
                    nEp, height(bsEpoch));

            % (e) artifact flag per epoch (1 = rejected by pop_artmwppth)
            rejFlag = EEG.reject.rejmanual(:);
            if isempty(rejFlag), rejFlag = zeros(nEp,1); end

            % (f) keep only artifact-accepted epochs. Their behavioral rows remain
            % aligned because pop_rejepoch preserves the order of surviving epochs.
            keep = (rejFlag==0);
            M = bsEpoch(keep, {'trialId','blockId','episode','triggerCode1','stimId', ...
                           'stimImage','itemType','itSRProb','bkSRProb','trialType', ...
                           'task','congruency','congruency_int','sbjCorr','ER','RT', ...
                           'directRep','voluntary'});
            % pop_rejepoch preserves order but renumbers the saved epochs 1..N.
            M.epoch_index = (1:height(M))';       % 1-based index in the SAVED .set
            M = movevars(M, 'epoch_index', 'Before', 'trialId');
            writetable(M, [metaDir 'S' num2str(S) '_metadata.csv']);
            fprintf('  metadata: %d accepted epochs aligned & written\n', height(M));
        end
    end

    % ---- ERP average (Criterion 'good') BEFORE removing epochs ----
    ERP = pop_averager(EEG, 'Criterion','good', 'DQ_custom_wins',0, 'DQ_flag',1, ...
              'DQ_preavg_txt',0, 'ExcludeBoundary','on', 'SEM','on');
    ERP = pop_savemyerp(ERP, 'erpname',['S' num2str(S) '_BP'], ...
              'filename',['S' num2str(S) '_BP.erp'], 'filepath',erpDir, 'Warning','off');
    pop_summary_AR_eeg_detection(EEG, [erpDir 'S' num2str(S) '_artifactDetection.txt']);

    % ---- physically remove flagged epochs, save clean set for MNE ----
    rejIdx = find(EEG.reject.rejmanual);
    fprintf('  removing %d flagged epochs (%d -> %d)\n', numel(rejIdx), EEG.trials, EEG.trials-numel(rejIdx));
    if ~isempty(rejIdx), EEG = pop_rejepoch(EEG, rejIdx, 0); end
    EEG.setname = ['S' num2str(S) '_preprocessed'];
    EEG = pop_saveset(EEG, 'filename',[EEG.setname '.set'], 'filepath',ppDir);

    close all; clear EEG ERP origEEG;
end

fprintf('\nDone. Clean sets in %s, metadata in %s\n', ppDir, metaDir);

% ==============================================================================
% helpers: robust event-code and epoch-label parsing
% ==============================================================================
function c = local_codes(EEG)
    c = local_codes_struct(EEG.event);
end

function c = local_codes_struct(ev)
    n = numel(ev); c = nan(1,n);
    for i = 1:n
        c(i) = local_scalar_num(ev(i).type);
    end
end

function a = local_merge_seam_items(EEG, itemCodes)
% Count item cues before a MID-BLOCK merge seam (from merged recording files).
% A merge seam is a boundary that (a) is NOT followed by a code-9 block marker
% AND (b) has item cues on BOTH sides. The both-sides test is essential: it
% excludes the recording-start boundary (0 item cues before it) and the
% recording-end boundary (0 after), either of which would otherwise be picked
% first and return the wrong position (e.g. 0). Returns the item-cue count
% before the seam (= the gap start index), or [] if there is no such seam.
    n = numel(EEG.event);
    isB = false(1,n); code = nan(1,n);
    for i = 1:n
        t = EEG.event(i).type;
        if (ischar(t)||isstring(t)) && strcmpi(char(t),'boundary')
            isB(i) = true;
        else
            code(i) = local_scalar_num(t);
        end
    end
    isItem = ismember(code, itemCodes);        % NaN-safe: non-items map to false
    a = [];
    for i = find(isB)
        j = i + 1;
        while j <= n && isB(j), j = j + 1; end  % skip consecutive boundaries
        nextCode = NaN; if j <= n, nextCode = code(j); end
        before = sum(isItem(1:i-1));
        after  = sum(isItem(i+1:end));
        if nextCode ~= 9 && before > 0 && after > 0
            a = before;                         % item cues before seam = gap start
            return;
        end
    end
end

function [rows, msg] = local_gap_align(E, L, aHint)
% Locate the single contiguous chunk of L (behavior) missing from E (EEG) by
% exact 1/2/3 code match. When several gap positions match (low-alphabet block),
% accept the merge-seam hint aHint only if it is itself among the verified
% candidates. Returns indices of L corresponding to E, or [] if unresolved.
    if nargin < 3, aHint = []; end
    nE = numel(E); nL = numel(L); gap = nL - nE;
    hits = [];
    for a = 0:nE
        b = a + gap + 1;                       % keep L(1:a), then L(b:nL)
        if b > nL + 1, continue; end
        cand = [L(1:a), L(b:nL)];
        if numel(cand) == nE && isequal(cand, E)
            hits(end+1) = a; %#ok<AGROW>
        end
    end
    extra = '';
    if isscalar(hits)
        a = hits(1);
    elseif ~isempty(hits) && ~isempty(aHint) && ismember(aHint, hits)
        a = aHint;
        extra = sprintf(' [%d candidates; merge seam selected a=%d]', numel(hits), a);
    elseif isempty(hits)
        rows = []; msg = sprintf('no single-gap alignment for %d missing trials', gap);
        return;
    else
        rows = []; msg = sprintf('%d ambiguous gap alignments (candidates %s), no usable seam hint', ...
                                 numel(hits), mat2str(hits));
        return;
    end
    b = a + gap + 1;
    rows = [1:a, b:nL];
    msg = sprintf('gap-aligned: %d behavioral trials (rows %d-%d) absent from EEG%s', ...
                  gap, a+1, b-1, extra);
end

function x = local_scalar_num(v)
    % Convert the scalar numeric/string/cell formats EEGLAB may use.
    if iscell(v)
        if isempty(v), x = NaN; return; end
        v = v{1};
    end
    if isnumeric(v) || islogical(v)
        if isempty(v), x = NaN; else, x = double(v(1)); end
        return;
    end
    d = regexp(char(string(v)), '\d+', 'match', 'once');
    if isempty(d), x = NaN; else, x = str2double(d); end
end

function code = local_epoch_item_code(EEG, ep, itemCodes)
    % Return the original item trigger (1/2/3) for the event at epoch time zero.
    % ERPLAB commonly stores event labels such as "B1,3(S1)". Parsing the first
    % number would incorrectly return the BIN number, so explicitly extract S1,
    % S2, or S3 from the parenthesized source-event tag.
    code = NaN;

    lat = local_numeric_vector(EEG.epoch(ep).eventlatency);
    types = EEG.epoch(ep).eventtype;
    if iscell(types)
        % already one cell per event
    elseif isnumeric(types) || islogical(types)
        types = num2cell(types);
    else
        % A single character/string event label must remain intact.
        types = {types};
    end

    candidates = find(abs(lat) < 1e-6);
    if isempty(candidates)
        candidates = 1:numel(types);
    end

    % Prefer the time-zero event; only search all events as a fallback.
    searchSets = {candidates, setdiff(1:numel(types), candidates, 'stable')};
    for ss = 1:numel(searchSets)
        idxs = searchSets{ss};
        for jj = idxs
            c = local_source_item_code(types{jj});
            if ismember(c, itemCodes)
                code = c;
                return;
            end
        end
    end
end

function c = local_source_item_code(v)
    % Decode an ORIGINAL source item code without confusing it with an ERPLAB bin.
    c = NaN;
    if iscell(v)
        if isempty(v), return; end
        v = v{1};
    end

    if isnumeric(v) || islogical(v)
        if ~isempty(v) && ismember(double(v(1)), [1 2 3])
            c = double(v(1));
        end
        return;
    end

    txt = char(string(v));

    % Typical ERPLAB/MNE label: B1,3(S1)/4 or B7(S2).
    tok = regexp(txt, '\(S([123])\)', 'tokens', 'once');
    if isempty(tok)
        % Also tolerate a plain S1/S2/S3 label.
        tok = regexp(txt, '(?:^|[^A-Za-z0-9])S([123])(?:$|[^0-9])', ...
                     'tokens', 'once');
    end
    if ~isempty(tok)
        c = str2double(tok{1});
        return;
    end

    % Only accept a bare numeric string; never use the first digit of B7(...).
    if ~isempty(regexp(strtrim(txt), '^[123]$', 'once'))
        c = str2double(strtrim(txt));
    end
end

function x = local_numeric_vector(v)
    % Convert EEG.epoch(...).eventlatency to a numeric row vector.
    if ~iscell(v)
        x = double(v(:)');
        return;
    end
    x = nan(1, numel(v));
    for ii = 1:numel(v)
        q = v{ii};
        if isnumeric(q) || islogical(q)
            if ~isempty(q), x(ii) = double(q(1)); end
        else
            x(ii) = str2double(string(q));
        end
    end
end
