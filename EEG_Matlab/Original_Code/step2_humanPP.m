
%% HUMAN INTERVENTION %%%%%%%%
% open up estudio and examine the xx_ICAv1.set, see which IC are blinks/saccades
%estudio
rootDir = 'C:\0_Projects\ISSC_wCue_E3\'
rawDir  = [rootDir 'raw\'];
preICADir = [rootDir 'preICA\'];
ppDir = [rootDir 'pp\'];

%% now the following chunck can be copy-paste and run in the command window

eyeIC ={[2 4],[3 10],[2 8],[1 7],[3 7],[1],[],[1 7],[],[],...
    [1 12],[9],[1 3],[5 10],[1],[1 4],[4],[10]}

for S=[1:18]
    EEG = pop_loadset('filename',['S' num2str(S) '_ICAv1.set'],'filepath',ppDir);
    EEG = pop_subcomp( EEG, eyeIC{S}, 0);
    [EEG,indelec] = pop_rejchan(EEG, 'elec',[1:EEG.nbchan] ,'threshold',5,'norm','on','measure','kurt');
    %load('bad_chan_idx.mat')
    indelec(indelec==1) =[];  % don't exclude Fp1
    indelec(indelec==31)=[];  % don't exclude Fp2
    bad_chan_idx{S} = indelec;
    clear EEG;

    % run ICA for the 2nd time without bad channels (identified after 1st attempt removing blinks, saccades)
    % ## if bad channels are Fp1 and Fp2, don't exclude those channels
    EEG  = pop_loadset('filename', ['S' num2str(S) '_preICA.set'] ,'filepath',preICADir);
    EEG2 = EEG;
    chanidx = [1:EEG.nbchan]; % all channels, no EOG
    chanidx(bad_chan_idx{S})=[];
    EEG2 = pop_resample(EEG2, 100);
    % you can further high-pass filter at 2 hz... also helps ICA decomposition
    EEG2 = pop_eegfiltnew(EEG2, 'locutoff', 2,'plotfreqz',1);
    EEG2 = pop_runica(EEG2, 'icatype', 'runica','extended',1,'rndreset','yes','chanind',chanidx);
    EEG2 = pop_iclabel(EEG2, 'default');

    % transfer weights to the original preICA.set, with full channels
    EEG = pop_editset(EEG, 'setname', 'postICA', 'icaweights', 'EEG2.icaweights', 'icasphere', 'EEG2.icasphere', 'icachansind', 'EEG2.icachansind');
    EEG.setname = 'postICA';
    EEG = pop_saveset( EEG, 'filename', ['S' num2str(S) '_' EEG.setname '.set'] ,'filepath',ppDir);
    t =['Run ICA 2nd time with ' num2str(size(chanidx,2)) ' channels, weights in EEG2, transfered to EEG/preICA.set']
    close('all')
    clear EEG EEG2
end

save('bad_chan_idx_S1-18.mat','bad_chan_idx')


return



%% HUMAN INTERVENTION %%%%%%%% (repeat...)
% open up estudio and examine the xx_postICA.set, see which IC are blinks/saccades
eyeIC_v2 ={ [1 2],[1 2],[3 6],[1 2],[1 2],...
            [2],[1 5],[1 2],[1 2],[1 4 11],...
            [1 6],[1 3],[2 11],[1 4],[1 2],...
            [1],[1 4],[1 2],[1 7],[2 7],...
            [1 5],[1 2],[6],[3],[6],[1 11]...
           }

% this is determined after estudio eyeballing
save('eyeICv2.mat',"eyeIC_v2");

% now the following chunck can be copy-paste and run in the command window
for S=[23]
    EEG  = pop_loadset('filename', ['S' num2str(S) '_postICA.set'] ,'filepath',ppDir);
    EEG = pop_subcomp( EEG, eyeIC_v2{S}, 0);
    EEG.setname = 'ICApruned';
    EEG = pop_saveset( EEG, 'filename', ['S' num2str(S) '_' EEG.setname '.set'] ,'filepath',ppDir);
    clear EEG;
end