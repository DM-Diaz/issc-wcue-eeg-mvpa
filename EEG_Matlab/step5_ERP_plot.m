% step5_ERP_plot.m -- 6-bin reward design
clear;
% cd('/home/moon/matlabToolbox/eeglab2025.0.0'); eeglab;

rootDir   = '/home/moon/Desktop/Dylan_Scripts_Summer_ISSC_wCue/EEG_Matlab/Data/All_Participants';
erpDir       = [rootDir 'erp_BP_test' filesep];
erpResultDir = [rootDir 'erp_BP_results_test' filesep];
cd(erpResultDir)

load('gpERP_filt_allbins.mat')   % gpERP, idx, binInfo, sbjList, chanlabels, chanlocs, time_vector
num_subs = length(sbjList);      % sbjList from step4 is already the final, exclusion-applied list
idx
size(gpERP)

% ---- pick the contrast ----
%  1 rpCond            2 swCond
%  3 rpCond-switch     4 rpCond-repeat
%  5 swCond-switch     6 swCond-repeat
%  7 all-switch        8 all-repeat        9 switch-minus-repeat
% 10 rpCond sw-effect 11 swCond sw-effect 12 sw-effect swCond-minus-rpCond
% 13 swCond-minus-rpCond

% bin_of_interest = [1 2 13];   figure_fn = ['ERP_rpCond_vs_swCond_N' num2str(num_subs)];
% bin_of_interest = [7 8 9];    figure_fn = ['ERP_switch_vs_repeat_N' num2str(num_subs)];
bin_of_interest = [10 11 12]; figure_fn = ['ERP_switchEffect_by_reward_N' num2str(num_subs)];

ERPlabel = arrayfun(@(b) binInfo(b).description, bin_of_interest, 'UniformOutput', false);

% ---- electrodes ----
legend_labels = {'Fz','Cz','Pz','Oz'};
[~, electrode_indices] = ismember(legend_labels, {chanlocs.labels});
if any(electrode_indices==0)
    error('electrode(s) not found: %s', strjoin(legend_labels(electrode_indices==0), ', '));
end
n_elecs = numel(legend_labels);

% ---- reshape: subject x time x bin x electrode ----
gpERP = permute(gpERP, [idx.sbj idx.time idx.bin idx.electrode]);

num_bins   = numel(bin_of_interest);
bin_colors = lines(num_bins);
tv = time_vector(:).';                       % force row for the fill() concatenation

figure;
nCol = 4;  nRow = ceil(n_elecs/nCol);
for eCNT = 1:n_elecs
    e = electrode_indices(eCNT);
    subplot(nRow, nCol, eCNT);  hold on;

    for b = 1:num_bins
        gpData = gpERP(:,:,bin_of_interest(b),e);
        M   = mean(gpData,1);
        SEM = std(gpData,0,1)/sqrt(num_subs);

        fill([tv, fliplr(tv)], [M+SEM, fliplr(M-SEM)], bin_colors(b,:), ...
             'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        h(b) = plot(tv, M, 'Color', bin_colors(b,:), 'LineWidth', 2);
    end

    yline(0, 'k:');  xline(0, 'k:');
    xlim([-200 800]);
    % difference waves are small; comment out the fixed ylim if they look flat
    ylim([-8 10]);
    title(legend_labels{eCNT});
    if eCNT == 1
        ylabel('Amplitude (\muV)');  xlabel('Time (ms)');
    else
        set(gca, 'YTickLabel', []);
    end
end

legend(h, ERPlabel, 'Location', 'northeast', 'Interpreter', 'none');
set(gcf, 'Position', [100 100 1200 300]);
print(gcf, [erpResultDir figure_fn '.eps'], '-depsc2', '-r300');
saveas(gcf, [erpResultDir figure_fn '.png']);
