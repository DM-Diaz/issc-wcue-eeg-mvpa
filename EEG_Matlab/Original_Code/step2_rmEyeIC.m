%% HUMAN INTERVENTION %%%%%%%% (repeat...)
% open up estudio and examine the xx_ICAv1.set, see which IC are blinks/saccades
% estudio
rootDir = 'C:\0_Projects\ISSC_wCue_E3\'

rawDir    = [rootDir 'raw' filesep];
preICADir = [rootDir 'preICA' filesep];
ppDir     = [rootDir 'pp' filesep];

% need cleaning 7
% this is determined after estudio eyeballing
eyeIC ={[2 4],[3 10],[2 8],[1 7],[3 7],[1],[],[1 7],[],[],...
    [1 12],[9],[1 3],[5 10],[1],[1 4],[4],[10]}

  


% now the following chunck can be copy-paste and run in the command window
for S=[1:18]
    EEG   = pop_loadset('filename', ['S' num2str(S) '_preICA.set'] ,'filepath',preICADir);
    EEG2  = pop_loadset('filename', ['S' num2str(S) '_ICAv1.set'] ,'filepath',ppDir);

    EEG   = pop_editset(EEG, 'setname', 'postICA', 'icaweights', 'EEG2.icaweights', 'icasphere', 'EEG2.icasphere', 'icachansind', 'EEG2.icachansind');
    EEG = pop_subcomp( EEG, eyeIC{S}, 0);
    EEG.setname = 'ICApruned';
    EEG = pop_saveset( EEG, 'filename', ['S' num2str(S) '_' EEG.setname '.set'] ,'filepath',ppDir);
    clear EEG;
end

%pop_selectcomps(EEG, [1:10]);