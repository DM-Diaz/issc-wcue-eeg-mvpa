% step4_ERP_difference.m -- 6-bin reward design, blocks 1-5
clear;
% cd('/home/moon/matlabToolbox/eeglab2025.0.0'); eeglab;

rootDir   = '/home/moon/Desktop/Dylan_Scripts_Summer_ISSC_wCue/EEG_Matlab/Data/All_Participants';
erpDir       = [rootDir 'erp_BP_test' filesep];
erpResultDir = [rootDir 'erp_BP_results_test' filesep];
bdfFile      = [rootDir 'binlister_v2.txt'];
diffFile     = [rootDir 'diff_wave.txt'];
if ~exist(erpResultDir,'dir'), mkdir(erpResultDir); end
cd(erpDir)

% FINAL list: hard exclusions (34, 50, 61, 64, 67) removed.
% Add further exclusions here if the MVPA uses the granular bins 3-6.
sbjList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
           11, 12, 13, 14, 15, 16, 17, 18, 20, ...
           21, 22, 23, 24, 25, 26, 27, 28, 29, ...
           31, 33, 35, 36, 37, 38, 40, ...
           41, 42, 43, 45, 46, 47, 48, 49, ...
           51, 52, 53, 54, 55, 56, 57, 58, 59, 60, ...
           65, 66, 68, 69, 70];

makeDerivedBins = true;   % false = keep only binlister's 6 bins, skip binoperator

% bins from binlister_v2.txt:
%   1 rpCond (lowSR)      2 swCond (highSR)
%   3 rpCond-switch       4 rpCond-repeat
%   5 swCond-switch       6 swCond-repeat
% derived bins 7-13 come from diff_wave.txt

sbjCNT = 0;
for S = sbjList
    sbjCNT = sbjCNT + 1;
    ERP = pop_loaderp('filename', ['S' num2str(S) '_BP.erp'], 'filepath', erpDir);

    if makeDerivedBins
        ERP = pop_binoperator(ERP, diffFile);
    end

    % NOTE: step1 already band-passed 0.2-20 Hz, so a 20 Hz low-pass here is a
    % no-op. Uncomment with a LOWER cutoff (e.g. 12) only for smoother display.
    % ERP = pop_filterp(ERP, [], 'Cutoff', 12, 'Design', 'butter', 'Filter', 'lowpass', 'Order', 8);

    erpname = ['S' num2str(S) '_diffwaves'];
    ERP = pop_savemyerp(ERP, 'erpname', erpname, 'filename', [erpname '.erp'], ...
                        'filepath', erpDir, 'Warning', 'off');
    gpERP(:,:,:,sbjCNT) = ERP.bindata;   % electrodes x time x bin x subject
end

idx.electrode = 1;  idx.time = 2;  idx.bin = 3;  idx.sbj = 4;

% bin labels: 1-6 from binlister, 7+ from diff_wave
binInfo = struct;
L = regexp(fileread(bdfFile), '\r\n|\n', 'split');
for i = 1:numel(L)-1
    if startsWith(strtrim(L{i}), 'bin ')
        binInfo(sscanf(strtrim(L{i}), 'bin %d')).description = strtrim(L{i+1});
    end
end
if makeDerivedBins
    L = regexp(fileread(diffFile), '\r\n|\n', 'split');
    for i = 1:numel(L)
        t = regexp(strtrim(L{i}), '^b(\d+)\s*=\s*(.+?)\s*label\s*(.+)$', 'tokens', 'once');
        if ~isempty(t)
            binInfo(str2double(t{1})).expression  = t{2};
            binInfo(str2double(t{1})).description = t{3};
        end
    end
end

% no EOG channel in this dataset (dropped at step1); 64 channels incl. reconstructed Cz
chanlocs    = ERP.chanlocs;
chanlabels  = {ERP.chanlocs.labels};
time_vector = ERP.times;

cd(erpResultDir)
save('gpERP_filt_allbins.mat','gpERP','idx','binInfo','sbjList','chanlabels','chanlocs','time_vector');
size(gpERP)
idx
