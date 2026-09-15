% step 2: load  preICA

rootDir = 'C:\0_Projects\ISSC_wCue_E3\'
rawDir  = [rootDir 'raw\'];
preICADir = [rootDir 'preICA\'];
ppDir = [rootDir 'pp\'];


for S=1:29

    EEG = pop_loadset('filename',['S' num2str(S) '_preICA.set'],'filepath',preICADir);
    EEG2=EEG;
    [EEG,indelec] = pop_rejchan(EEG, 'elec',[1:EEG.nbchan] ,'threshold',5,'norm','on','measure','kurt');
    
    indelec(indelec==1) =[];  % don't exclude Fp1
    indelec(indelec==31)=[];  % don't exclude Fp2

    
    chanidx = [1:EEG2.nbchan]; % all channels, no EOG
    chanidx(indelec)=[];

    EEG2 = pop_resample(EEG2, 100); % faster process
    EEG2 = pop_eegfiltnew(EEG2, 'locutoff', 2,'plotfreqz',1);
    EEG2 = pop_runica(EEG2, 'icatype', 'runica','extended',1,'rndreset','yes','chanind',chanidx); % only run on good chanels
    EEG2 = pop_iclabel(EEG2, 'default');
    EEG2.setname = 'ICAv2';
    EEG2 = pop_saveset( EEG2, 'filename', ['S' num2str(S) '_' EEG.setname '.set'] ,'filepath',ppDir);
    close all;clear EEG;
end

