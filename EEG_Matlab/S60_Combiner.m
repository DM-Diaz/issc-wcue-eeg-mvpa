%% ===== Combine S60 Part 1 + Part 2 into S60_ICApruned.set =====
ppDir = '/home/moon/Desktop/ISSC_wCue/pp/';
f1 = 'S60_ICApruned.set';           % Part 1
f2 = 'S60 part 2_ICApruned.set';    % Part 2  (confirm exact name: dir([ppDir 'S60*']))

E1 = pop_loadset('filename', f1, 'filepath', ppDir);
E2 = pop_loadset('filename', f2, 'filepath', ppDir);

% 1. channels must match to merge cleanly
if ~isequal({E1.chanlocs.labels}, {E2.chanlocs.labels})
    fprintf(2,'only in Part1: %s\n', strjoin(setdiff({E1.chanlocs.labels},{E2.chanlocs.labels}),', '));
    fprintf(2,'only in Part2: %s\n', strjoin(setdiff({E2.chanlocs.labels},{E1.chanlocs.labels}),', '));
    error('S60 parts have different channels; resolve before merging (see note below).');
end
fprintf('Channels match (%d). OK to merge.\n', E1.nbchan);

% 2. confirm Part1 is chronologically first
if ~isempty(E1.event(1).bvtime) && ~isempty(E2.event(1).bvtime) && E2.event(1).bvtime < E1.event(1).bvtime
    warning('Part2 timestamp precedes Part1; check which file is which.');
end

% 3. clear ICA fields (already pruned; avoids eeg_checkset recomputing on the merge)
for c = {'icaweights','icasphere','icawinv','icaact','icachansind'}
    E1.(c{1}) = [];  E2.(c{1}) = [];
end

% 4. merge (Part1 then Part2; a boundary is inserted at the junction)
EEG = eeg_checkset(pop_mergeset(E1, E2, 0));

% 5. sanity: expect 7 markers, block 5 = 74 trial cues
codes = nan(1,numel(EEG.event));
for i=1:numel(EEG.event)
    t=EEG.event(i).type;
    if isnumeric(t), codes(i)=double(t);
    else, d=regexp(char(t),'\d+','match','once'); if ~isempty(d), codes(i)=str2double(d); end
    end
end
blk = find(codes==9);  bounds=[blk numel(codes)+1];
fprintf('Merged S60: %d block markers (expect 7)\n', numel(blk));
for b=1:numel(bounds)-1
    fprintf('  block %d: %d trial cues\n', b, sum(ismember(codes(bounds(b):bounds(b+1)-1),[1 2 3])));
end

% 6. back up both original parts (same names, into a subfolder to preserve .set/.fdt linkage),
%    then save the merged set under the name step3 loads
bkDir = [ppDir 'backup_S60' filesep];  if ~exist(bkDir,'dir'), mkdir(bkDir); end
for nm = {f1, f2}
    movefile([ppDir nm{1}], [bkDir nm{1}]);
    fdt = strrep(nm{1},'.set','.fdt');
    if isfile([ppDir fdt]), movefile([ppDir fdt], [bkDir fdt]); end
end
EEG.setname = 'S60_ICApruned';
EEG = pop_saveset(EEG, 'filename', 'S60_ICApruned.set', 'filepath', ppDir);
fprintf('Saved merged S60_ICApruned.set; originals in backup_S60/.\n');

