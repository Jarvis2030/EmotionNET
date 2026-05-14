clear;

%% Parameter defintion
fs = 128;
total_subs = 23;
total_videos = 18;
total_channels = 14;

seg_win = 3; % 3s segmentation

%% restructure data

DB = load("DREAMER.mat").DREAMER.Data;

stim_subject_data = cell(18,23);
base_subject_data = cell(18,23);

for sub = 1:total_subs
    for clip = 1:total_videos
        
        % --- EEG ---
        stim = DB{sub}.EEG.stimuli(clip);
        base = DB{sub}.EEG.baseline(clip);
        eeg_data_s = stim{1};
        eeg_data_b = base{1};% (14 × M)
        
        % --- scores ---
        val = DB{sub}.ScoreValence(clip);
        aro = DB{sub}.ScoreArousal(clip);
        dom = DB{sub}.ScoreDominance(clip);
        
        % --- 存進 cell ---
        temp.seeg = eeg_data_s;
        temp.beeg = eeg_data_b;
        temp.valence = val;
        temp.arousal = aro;
        temp.dominance = dom;
        
        subject_data{clip, sub} = temp;
    end
end
% baseline removal

% Z-score Normalization


%% Data Cleaning

% fetch data & detrend
FeatureTable = table();

for sub = 1:total_subs
    for vid = 1:total_videos
        for ch = 1:total_channels

            % Data cleaning
            [stim_filtered, label_val, label_aro] = ...
                AllFunctions.data_cleaning(subject_data, ch, sub, vid, fs);

            % Segmentation
            stim_seg = AllFunctions.segmentation(stim_filtered, seg_win, fs);
            num_seg = size(stim_seg,2);

            % Loop each segment
            for seg = 1:num_seg

                eeg_data = stim_seg(:,seg);
                eeg_band = AllFunctions.EEG_band_generate(eeg_data, 0, fs);

                % feature extraction 
                de = zeros(1,5);
                psd = zeros(1,5);

                for j = 1:5
                    de(j) = 0.5 * log(2*pi*exp(1)*var(eeg_band(j,:)));
                    psd(j) = log10(mean(eeg_band(j,:).^2) + eps);
                end

                hfd = AllFunctions.HFD(eeg_data);

                % Create one row
 
                new_row = table( ...
                    sub, vid, ch, seg, ...
                    de(1), de(2), de(3), de(4), de(5), ...
                    psd(1), psd(2), psd(3), psd(4), psd(5), ...
                    hfd, label_val, label_aro, ...
                    'VariableNames', { ...
                    'subject','video','channel','segment', ...
                    'DE_delta','DE_theta','DE_alpha','DE_beta','DE_gamma', ...
                    'PSD_delta','PSD_theta','PSD_alpha','PSD_beta','PSD_gamma', ...
                    'HFD','label_val','label_aro'} ...
                );

                FeatureTable = [FeatureTable; new_row];

            end
        end
    end
end

save('EEG_FeatureTable.mat','FeatureTable','-v7.3')
writetable(FeatureTable, 'EEG_Features.csv')

%%

% ---------------- Phase checking --------------
% subplot(1, 2, 1); 
% plot(stim);
% title('Original Stimulus');
% xlabel('Time');
% ylabel('Amplitude');axis tight;
% 
% subplot(1, 2, 2); 
% plot(stim_filtered);
% title('Filtered Stimulus');
% xlabel('Time');
% ylabel('Amplitude'); axis tight;
% % ---------------------------------------------


%% Statistic Features
% Mean
FeatureTable.mean_power = mean(table2array(FeatureTable(:,5:end-2)), 2); % 有問題之後處理 mean formula 是錯的
% lag Variable
FeatureTable = AllFunctions.add_lag_features(FeatureTable, 3, seg_win); % lag 1, lag 2

%% move the label back to last 2 columns
vars = FeatureTable.Properties.VariableNames;

% 找 label
label_vars = {'label_val','label_aro'};

% 其他欄位（排除 label）
other_vars = setdiff(vars, label_vars, 'stable');

% 重排
FeatureTable = FeatureTable(:, [other_vars, label_vars]);

remove_vars = vars(contains(vars, 'label_val_'));
FeatureTable(:, remove_vars) = [];

remove_vars = vars(contains(vars, 'label_aro_'));
FeatureTable(:, remove_vars) = [];

remove_vars = vars(contains(vars, 'PSD_'));
FeatureTable(:, remove_vars) = [];

%% Signal Analysis (FeatureTable(:,5:end-2) remove IDs and Label
vars = FeatureTable.Properties.VariableNames;

% feature columns remove metadata and label
feature_idx = 5:(width(FeatureTable)-2);

X = table2array(FeatureTable(:, feature_idx));
feature_names = vars(feature_idx);

X = table2array(FeatureTable(:,5:end-2)); % features
y_val = FeatureTable.label_val;
y_aro = FeatureTable.label_aro;

% feature distribution to check if normalization is needed
figure;
t = tiledlayout(7,7,'TileSpacing','compact');
set(gca,'TickLabelInterpreter','none');
for i = 1:size(X,2)
    nexttile;
    histogram(X(:,i))
    title(FeatureTable.Properties.VariableNames{i+4},'FontSize',20);
end


X = table2array(FeatureTable(:,5:end-2));
Y_val = FeatureTable.label_val;
Y_aro = FeatureTable.label_aro;

corr_vals = corr(X, Y_val, 'Type','Spearman');
corr_arou = corr(X, Y_aro, 'Type','Spearman');
R = corr(X, 'Type','Spearman');

[~, idx] = sort(abs(corr_vals), 'descend');
top10 = idx(1:10);

figure;
set(gca,'FontSize',16);
bar(corr_vals)
title('Feature vs Valence Correlation','FontSize',20)
xticks(1:length(feature_names))
xticklabels(FeatureTable.Properties.VariableNames(5:end-2));
xtickangle(45)
xtickformat( ...
    )

figure;
set(gca,'FontSize',16);
bar(corr_arou)
title('Feature vs Arousal Correlation','FontSize',20)
xticks(1:length(feature_names))
xticklabels(FeatureTable.Properties.VariableNames(5:end-2));
xtickangle(45)

figure;
h = imagesc(R);
set(gca,'TickLabelInterpreter','none');
set(gca,'FontSize',16);
colorbar;
title('Feature Correlation Matrix','FontSize',20);

xticks(1:length(feature_names));
yticks(1:length(feature_names));

xticklabels(feature_names);
yticklabels(feature_names);

xtickangle(90);

% anova test val
p_values_val = zeros(1,size(X,2));

for i = 1:size(X,2)
    p = anova1(X(:,i), y_val, 'off');
    p_values_val(i) = p;
end
figure;
set(gca,'FontSize',16);
bar(-log10(p_values_val)); 

hold on
yline(-log10(0.05), 'r--', 'Threshold (p=0.05)','FontSize',16)

title('ANOVA (Valence)''FontSize',24)
xticks(1:length(feature_names))
xticklabels(feature_names)
xtickangle(45)

set(gca,'TickLabelInterpreter','none')
ylabel('-log10(p)')

% anova test arou
p_values_aro = zeros(1,size(X,2));

for i = 1:size(X,2)
    p = anova1(X(:,i), y_aro, 'off');
    p_values_aro(i) = p;
end
figure;
set(gca,'FontSize',16);
bar(-log10(p_values_aro))

hold on
yline(-log10(0.05), 'r--')

title('ANOVA (Arousal)');
xticks(1:length(feature_names))
xticklabels(feature_names)
xtickangle(45)

set(gca,'TickLabelInterpreter','none')
set(gca,'FontSize',16);
ylabel('-log10(p)')


