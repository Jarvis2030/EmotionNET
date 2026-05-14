
classdef AllFunctions
    methods(Static)
        function [clean_data, label_val,label_aro] = data_cleaning(subject_data,ch,sub,vid,fs)
            stim = subject_data{vid,sub}.seeg(:,ch)- mean(subject_data{vid,sub}.seeg(:,ch));
            base = subject_data{vid,sub}.beeg(:,ch)- mean(subject_data{vid,sub}.beeg(:,ch));
            
            val = subject_data{vid,sub}.valence;
            aro = subject_data{vid,sub}.arousal;
            
            % filtering
            stim_filtered = bandpass(stim,[0.5 45],fs);
            
            Wo = 60/(fs/2);  
            BW = Wo/35;% Design notch filter with a Q-factor of Q=35
            [b,a] = iirnotch(Wo,BW); 
            stim_filtered = filter(b,a,stim_filtered);
            
            % baseline removal 
            base_mean = mean(base);
            clean_data = stim_filtered - base_mean;
            
            % label transformation
            if val <= 3
                label_val = 0;   % negative
            elseif val <= 6
                label_val = 1;   % neutral
            else
                label_val = 2;   % positive
            end

             if aro <= 3
                label_aro = 0;   % negative
            elseif val <= 6
                label_aro = 1;   % neutral
            else
                label_aro = 2;   % positive
            end


        end

        function stim_seg = segmentation(stim_filtered, seg_window, fs)
            
            % default segmentaiton length = 3s
            if nargin < 2
                seg_window = 3;
            end
 
            remainder = mod(length(stim_filtered), seg_window * fs);

            % split trimming evenly (front + end)
            trim_front = floor(remainder/2);
            trim_back  = remainder - trim_front;
        
            start_idx = 1 + trim_front;
            end_idx   = length(stim_filtered) - trim_back;
        
            stim_trimmed = stim_filtered(start_idx:end_idx);
            stim_seg = reshape(stim_trimmed, seg_window * fs, []);
        end

        function band = EEG_band_generate(eeg_data, plot_graph, fs)
          % extracting the details and approximation coefficients 
          % delta: 0.5–4Hz, theta: 4–8Hz, alpha: 8–13Hz, beta: 13–30Hz, gamma: >30Hz 

        % default not to plot graph
        if nargin < 2
            plot_graph = false;
        end
    
        % seperate into frequency band usiong DFT
        waveletFunction = 'db8';
        [C,L] = wavedec(eeg_data,8,waveletFunction);
     
        
        % Reconstructing the Signal components
        gamma = wrcoef('d',C,L,waveletFunction,5); 
        beta = wrcoef('d',C,L,waveletFunction,6); 
        alpha = wrcoef('d',C,L,waveletFunction,7); 
        theta = wrcoef('d',C,L,waveletFunction,8); 
        delta = wrcoef('a',C,L,waveletFunction,8); 
        
        theta = detrend(theta,0);
        
        all_bands = {eeg_data, gamma, beta, alpha, theta, delta};
        min_len = min(cellfun(@length, all_bands)); % find min length to avoid Index out of bounds
        
        % trimmed to minimum length
        gamma = gamma(1:min_len);
        beta  = beta(1:min_len);
        alpha = alpha(1:min_len);
        theta = theta(1:min_len);
        delta = delta(1:min_len);
        sig_plot = eeg_data(1:min_len);

        band = [delta'; theta'; alpha'; beta'; gamma'];
    
        if plot_graph
    
            time_axis = (0:min_len-1) / fs;
            
            % plotting the frequency bands
            figure('Color', 'w');
            t = tiledlayout(6, 1, 'Padding', 'compact', 'TileSpacing', 'tight');
            
            titles = {'Original (4-45Hz)', 'GAMMA (30-45Hz)', 'BETA (13-30Hz)', ...
                      'ALPHA (8-13Hz)', 'THETA (4-8Hz)', 'DELTA (1-4Hz)'};
            data_plot = {sig_plot, gamma, beta, alpha, theta, delta};
            
            for i = 1:6
                ax = nexttile;
                plot(time_axis, data_plot{i}, 'Color', '#0072BD');
                title(titles{i}, 'FontSize', 10);
                
                ylabel('Amp (\muV)');
                grid on;
               
                xlim([0, max(time_axis)]); 
                
                if i < 6
                    xticklabels({});
                else
                    xlabel('Time (seconds)');
                end
            end
            
            % Big title
            title(t, 'EEG Frequency Bands Decomposition', 'FontSize', 12, 'FontWeight', 'bold');
        end
        end

        function hfd_features = HFD(eeg_data)
            % all the frequency band
            kmax = 10; % 對於 128Hz 採樣率，kmax 建議設為 6~10
            N = length(eeg_data);
            
            L = zeros(1, kmax);
            for k = 1:kmax
                Lm = zeros(1, k);
                for m = 1:k
                    % 重構子序列並計算長度
                    n_max = floor((N - m) / k);
                    norm_factor = (N - 1) / (n_max * k);
                    
                    % 計算序列點對點的絕對差值總和
                    series_diff = abs(diff(eeg_data(m:k:m + n_max * k)));
                    Lm(m) = (sum(series_diff) * norm_factor) / k;
                end
                L(k) = mean(Lm); % 計算平均長度
            end
            
            % 線性擬合 ln(L) 對 ln(1/k) 的斜率
            % y = ax + b -> HFD 就是斜率 a
            p = polyfit(log(1./(1:kmax)), log(L), 1);
            hfd_features = p(1);
        end

        function T = add_lag_features(T, lag_steps, seg_window)

            feature_cols = 5:(width(T)-2);
            feature_names = T.Properties.VariableNames(feature_cols);
        
            % group by subject + video + channel
            groups = findgroups(T.subject, T.video, T.channel);

            
        
            for lag = 1:lag_steps
                lag_chunk = seg_window * lag;
                for f = 1:length(feature_names)
        
                    fname = feature_names{f};
        
                    lag_name  = [fname, '_lag', num2str(lag)];
                    diff_name = [fname, '_diff', num2str(lag)];
        
                    lag_col = nan(height(T),1);
        
                    for g = 1:max(groups)
                        idx = find(groups == g);
        
                        data = T{idx, fname};
        
                        if length(data) > lag_chunk
                            lagged = [nan(lag_chunk,1); data(1:end-lag_chunk)];
                        else
                            lagged = nan(length(data),1);
                        end
        
                        lag_col(idx) = lagged;
                    end
                    T.(lag_name) = lag_col;
                    T.(diff_name) = T.(fname) - T.(lag_name);
        
                end
            end

            T = rmmissing(T);
        
        end
    end
end