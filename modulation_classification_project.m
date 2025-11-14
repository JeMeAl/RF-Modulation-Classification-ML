%% ========================================================================
% RF Modulation Classification with Machine Learning (Software-Only)
% Author: Brian Rono
%
% This script generates several digital modulation schemes (BPSK, QPSK,
% 8PSK, 16QAM), passes them through an AWGN channel at different SNRs, then
% extracts simple RF-inspired features from the complex baseband signal and
% trains a multi-class classifier to recognize the modulation type.
%
% Dependencies:
%   - Base MATLAB
%   - Recommended: Communications Toolbox
%   - Recommended: Statistics & Machine Learning Toolbox
% ========================================================================

clear; close all; clc;

%% ------------------------ 1. Global Parameters --------------------------
modNames = {'BPSK','QPSK','8PSK','16QAM'};  % modulation types
Mvals    = [2      4       8      16   ];   % constellation sizes

Ns_per_example       = 256;                 % symbols in each example
Nexamples_per_class  = 100;                 % examples per modulation
SNRdB_range          = [5 15];              % training SNR range (dB)

Fs = 1e6;                                   % sampling rate for plotting
Ts = 1/Fs;

fprintf('Total training examples: %d\n', ...
    Nexamples_per_class*numel(modNames));

%% ------------------------ 2. Feature Matrix & Labels --------------------
Nclasses  = numel(modNames);
Nexamples = Nclasses * Nexamples_per_class;

% Feature vector: [var(I), var(Q), mean(|s|), mean(|s|^2), kurtosis(|s|)]
Nfeatures = 5;

X = zeros(Nexamples, Nfeatures);
Y = strings(Nexamples,1);

example_idx = 1;

%% ------------------------ 3. Generate Data & Features -------------------
for ci = 1:Nclasses
    M     = Mvals(ci);
    mName = modNames{ci};

    for n = 1:Nexamples_per_class

        % Random SNR for this example
        SNRdB = SNRdB_range(1) + ...
                (SNRdB_range(2)-SNRdB_range(1))*rand;

        % Random integer symbols
        data = randi([0 M-1], Ns_per_example, 1);

        % Baseband modulation
        s_tx = baseband_modulate(data, mName, M);

        % AWGN channel
        s_rx = awgn(s_tx, SNRdB, 'measured');

        % Extract RF-style features
        X(example_idx,:) = extract_features(s_rx);

        % Label
        Y(example_idx) = string(mName);

        example_idx = example_idx + 1;
    end
end

%% ------------------------ 4. Constellation Examples --------------------
figure;
tiledlayout(2,2,"Padding","compact","TileSpacing","compact");

for ci = 1:Nclasses
    M     = Mvals(ci);
    mName = modNames{ci};

    data_demo = randi([0 M-1], Ns_per_example, 1);
    s_demo    = baseband_modulate(data_demo, mName, M);
    s_demo    = awgn(s_demo, 15, 'measured');  % moderate SNR

    nexttile;
    plot(real(s_demo), imag(s_demo), '.');
    axis equal; grid on;
    title(sprintf('%s Constellation', mName));
    xlabel('In-Phase'); ylabel('Quadrature');
end

% saveas(gcf, 'q1.png');  % optional

%% ------------------------ 5. Time and Spectrum View (QPSK) -------------
M_QPSK = 4;
data_qpsk = randi([0 M_QPSK-1], Ns_per_example, 1);
s_qpsk    = baseband_modulate(data_qpsk, 'QPSK', M_QPSK);
s_qpsk    = awgn(s_qpsk, 10, 'measured');

Nfft  = 1024;
S_f   = fftshift(fft(s_qpsk, Nfft));
faxis = (-Nfft/2:Nfft/2-1)*(Fs/Nfft);

figure;
subplot(2,1,1);
plot((0:Ns_per_example-1)*Ts, abs(s_qpsk));
xlabel('Time (s)'); ylabel('|s(t)|');
title('QPSK Magnitude vs Time'); grid on;

subplot(2,1,2);
plot(faxis/1e3, 20*log10(abs(S_f)));
xlabel('Frequency (kHz)'); ylabel('Magnitude (dB)');
title('QPSK Spectrum (Magnitude)'); grid on;

% saveas(gcf, 'q2.png');  % optional

%% ------------------------ 6. Train/Test Split ---------------------------
rng(1);                                      % reproducible shuffle
idx         = randperm(Nexamples);
train_ratio = 0.7;
Ntrain      = round(train_ratio * Nexamples);

Xtrain = X(idx(1:Ntrain), :);
Ytrain = Y(idx(1:Ntrain), :);
Xtest  = X(idx(Ntrain+1:end), :);
Ytest  = Y(idx(Ntrain+1:end), :);

%% ------------------------ 7. Train Multi-Class Classifier ---------------
Mdl = fitcecoc(Xtrain, Ytrain);              % SVM-based ECOC model

Ypred = predict(Mdl, Xtest);
acc   = mean(Ypred == Ytest);

fprintf('Overall modulation classification accuracy: %.2f %%\n', acc*100);

%% ------------------------ 8. Feature Space & Confusion Matrix ----------
% Feature-space view: mean(|s|) vs kurtosis(|s|)
figure;
gscatter(X(:,3), X(:,5), Y);
xlabel('Mean magnitude');
ylabel('Kurtosis of magnitude');
title('Feature Space: Mean vs Kurtosis (per modulation)');
grid on;

% saveas(gcf, 'q3.png');  % optional

% Confusion matrix as Figure 4, saved as q4.png
figure;
confusionchart(categorical(Ytest), categorical(Ypred));
title('Modulation Classification Confusion Matrix');
grid on;
saveas(gcf, 'q4.png');

%% ========================================================================
% Helper Functions
% ========================================================================

function s = baseband_modulate(data, modName, M)
%BASEBAND_MODULATE Generate complex baseband symbols for a modulation type.
% data    : integer symbols 0...(M-1)
% modName : 'BPSK','QPSK','8PSK','16QAM'
% M       : constellation size

    modName = upper(string(modName));

    switch modName
        case "BPSK"
            % Map bits {0,1} to {+1,-1} on real axis
            s = 2*double(data) - 1;
            s = complex(s, 0);

        case "QPSK"
            s = pskmod(data, M, pi/4, 'gray');

        case "8PSK"
            s = pskmod(data, M, 0, 'gray');

        case "16QAM"
            s = qammod(data, M, 'gray', 'UnitAveragePower', true);

        otherwise
            error('Unsupported modulation type: %s', modName);
    end
end

function f = extract_features(s)
%EXTRACT_FEATURES Compute a small set of RF features from complex samples.
% s : complex baseband vector
%
% Features:
%   1) variance of I
%   2) variance of Q
%   3) mean magnitude
%   4) mean squared magnitude (power)
%   5) kurtosis of magnitude

    s = s(:);   % column vector

    I   = real(s);
    Q   = imag(s);
    mag = abs(s);

    var_I  = var(I);
    var_Q  = var(Q);
    mean_m = mean(mag);
    mean_p = mean(mag.^2);
    kurt_m = kurtosis(mag);

    f = [var_I, var_Q, mean_m, mean_p, kurt_m];
end
