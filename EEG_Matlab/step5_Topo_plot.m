% step5_Topo_plot.m -- 6-bin reward design
clear;
% cd('/home/moon/matlabToolbox/eeglab2025.0.0'); eeglab;

rootDir   = '/home/moon/Desktop/Dylan_Scripts_Summer_ISSC_wCue/EEG_Matlab/Data/All_Participants';
erpDir       = [rootDir 'erp_BP_test' filesep];
erpResultDir = [rootDir 'erp_BP_results_test' filesep];
cd(erpResultDir)

load('gpERP_filt_allbins.mat')   % gpERP (elec x time x bin x sbj), idx, binInfo, sbjList, chanlocs, time_vector
num_subs = numel(sbjList);       % sbjList from step4 is already the final list
idx
size(gpERP)

% ---- contrasts: {bins, filename, color limits} ----
%  1 rpCond            2 swCond
%  3 rpCond-switch     4 rpCond-repeat
%  5 swCond-switch     6 swCond-repeat
%  7 all-switch        8 all-repeat        9 switch-minus-repeat
% 10 rpCond sw-effect 11 swCond sw-effect 12 sw-effect swCond-minus-rpCond
% 13 swCond-minus-rpCond
contrasts = {
    [1 2],      'Topo_rpCond_vs_swCond',       [-10 10]   % raw ERPs
    [13],       'Topo_rewardDiff',             [-1.5 1.5]     % difference wave
    [7 8],      'Topo_switch_vs_repeat',       [-10 10]   % raw
    [9],        'Topo_transitionDiff',         [-1.5 1.5]     % difference wave
    [10 11 12], 'Topo_switchEffect_by_reward', [-1.5 1.5]     % all differences
};
% note raw bins color scale higher vs. diff bins color scale should be
% lower: raw [-10, 10] default vs. diff [-3, 3] default
% change depending on observed absolute amplitude value

time_windows = 0:100:600;
n_windows    = numel(time_windows);
cmap         = jet;                       

[~, chm] = ismember({'Fz','Cz','Pz','Oz'}, {chanlocs.labels});
chm = chm(chm > 0);                          % drop any not found

for c = 1:size(contrasts,1)
    bin_of_interest = contrasts{c,1};
    figure_fn       = [contrasts{c,2} '_N' num2str(num_subs)];
    clim            = contrasts{c,3};

    ERPlabel = arrayfun(@(b) binInfo(b).description, bin_of_interest, 'UniformOutput', false);
    nrows = numel(bin_of_interest);
    ncols = n_windows;

    figure;
    for b = 1:nrows
        for i = 1:n_windows
            t_start = time_windows(i);
            t_end   = t_start + 100;
            samples_idx = find(time_vector >= t_start & time_vector <= t_end);

            gpData  = gpERP(:, samples_idx, bin_of_interest(b), :);  % elec x time x 1 x sbj
            M       = mean(gpData, 4);                               % mean over subjects
            meanAmp = mean(M, 2);                                    % mean over the window

            subplot(nrows, ncols, (b-1)*ncols + i);
            topoplot(meanAmp, chanlocs, 'style','map', 'maplimits',clim, ...
                     'emarker',{'.',[.8 .8 .8],[],1}, 'emarker2',{chm,'o','k',3,1}, ...
                     'colormap',cmap);
            if b == 1
                title([num2str(t_start) '-' num2str(t_end) ' ms']);
            end
        end

        % row label, left margin
        ypos = 1 - b/nrows + 0.5/nrows - 0.05;
        annotation('textbox', [0.01 ypos 0.08 0.1], 'String', ERPlabel{b}, ...
                   'FitBoxToText','on', 'EdgeColor','none', 'FontWeight','bold', ...
                   'FontSize',11, 'HorizontalAlignment','center', ...
                   'VerticalAlignment','middle', 'Interpreter','none');
    end

    colormap(cmap);
    cb = colorbar('Position', [0.93 0.11 0.015 0.77]);
    cb.Label.String = 'Amplitude (\muV)';
    cb.FontSize = 14;
    caxis(clim);

    set(gcf, 'Position', [100 100 1600 200+220*nrows]);
    print(gcf, [erpResultDir figure_fn '.eps'], '-depsc2', '-painters', '-r300');
    saveas(gcf, [erpResultDir figure_fn '.png']);
    fprintf('wrote %s (%d rows)\n', figure_fn, nrows);
end
