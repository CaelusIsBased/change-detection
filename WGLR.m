%Window Limited GLR

%optimal window length w? as gamma -> inf, optimal w -> log(gamma).
%rng(42);

%d = 20;
gamma = 1000;
kappa = 100;
n_seq = 1000;
n_cal = 10;
n_edd = 1000;
%w = 42;
M = 2500;

% calibrating w
a = 2.0;  % tune this — controls detection speed vs false alarm tradeoff
b_gs = 1.1;  % geometric spacing ratio, Lai suggests close to 1
w_min = max(2, floor(a * log(gamma)));
w_max = gamma;  % or cap lower, e.g. floor(gamma^0.5) for computation

j = 0;
W_set = [1];
while true
    nj = floor(b_gs^j * w_min);
    if nj >= w_max; break; end
    W_set(end+1) = nj;
    j = j + 1;
end
W_set = unique(W_set);

% case 1
% d = 20;
% w1 = 7/8; mu1 = (1/4)*ones(d, 1);
% w2 = 1/8; mu2 = zeros(d, 1);

%case2
% d  = 50;
% w1 = 1/2; mu1 = (1/3)*ones(d,1);
% w2 = 1/2; mu2 = zeros(d,1);

%case 3
% sampling only — LLR unchanged
d = 20;
function X = sample_post(d, n)
    U = rand(d, n) - 0.5;
    X = 0.5 - 0.25 * sign(U) .* log(1 - 2*abs(U));
end

%case4
% d = 20;
% function X = sample_post(d, n)
%     X = -1 + (4/5) * (-log(rand(d,n)));
% end

%case5
% d = 20;
% function X = sample_post(d, n)
%     X = -0.5 + 2*rand(d,n);
% end

ref_data = randn(d, M);
mu_hat = mean(ref_data, 2);
diff_ref = ref_data - mu_hat;
Sigma_hat = (diff_ref * diff_ref') / (M - 1);
Sigma_inv = inv(Sigma_hat);

% function S = wglr_statistic(seq, t, w, mu_hat, Sigma_inv)
% S = -Inf;
% kappa_lo = max(1, t - w);
% kappa_hi = t - 1;
% 
% Y_centered = seq - mu_hat;
% cumsum_Y = cumsum(Y_centered, 2);
% 
% for kap = kappa_lo:kappa_hi
%     n_win = t - kap;
% 
%     if kap >= 1
%         window_sum = cumsum_Y(:, t) - cumsum_Y(:, kap);
%     else
%         window_sum = cumsum_Y(:, t);
%     end
% 
%     stat = (window_sum' * Sigma_inv * window_sum) / n_win;
%     if stat > S
%         S = stat;
%     end
% end
% end

% function T = run_wglr(seq, b, w, mu_hat, Sigma_inv)
% n = size(seq, 2);
% T = Inf;
% for t = 1:n
%     S = wglr_statistic(seq(:, 1:t), t, w, mu_hat, Sigma_inv);
%     if S > b
%         T = t;
%         return;
%     end
% end
% end

function S = wglr_statistic(seq, t, W_set, mu_hat, Sigma_inv)
S = -Inf;
valid_w = W_set(W_set < t);
if isempty(valid_w); return; end

Y_centered = seq - mu_hat;
cumsum_Y = cumsum(Y_centered, 2);

for w = valid_w
    kap = t - w;
    if kap >= 1
        window_sum = cumsum_Y(:, t) - cumsum_Y(:, kap);
    else
        window_sum = cumsum_Y(:, t);
    end
    stat = (window_sum' * Sigma_inv * window_sum) / w;
    if stat > S
        S = stat;
    end
end
end

function T = run_wglr(seq, b, W_set, mu_hat, Sigma_inv)
n = size(seq, 2);
T = Inf;
for t = 1:n
    S = wglr_statistic(seq(:, 1:t), t, W_set, mu_hat, Sigma_inv);
    if S > b
        T = t;
        return;
    end
end
end

function X = sample_mixture(d, n, w1, mu1, w2, mu2)
X = zeros(d, n);
for i = 1:n
    if rand < w1
        X(:, i) = mu1 + randn(d, 1);
    else
        X(:, i) = mu2 + randn(d, 1);
    end
end
end

fprintf('Threshold\n');
n_h0 = 10 * gamma;
b_lo = 0.5;
b_hi = 100.0;

for iter = 1:20
    b_mid = (b_lo + b_hi) / 2;
    stop_times = zeros(1, n_cal);
    for trial = 1:n_cal
        seq_h0 = randn(d, n_h0);
        %stop_times(trial) = run_wglr(seq_h0, b_mid, w, mu_hat, Sigma_inv);
        stop_times(trial) = run_wglr(seq_h0, b_mid, W_set, mu_hat, Sigma_inv);
    end
    stop_times(isinf(stop_times)) = n_h0;
    est_arl = mean(stop_times);

    fprintf('Iter %2d: b = %.4f, ARL = %.1f\n', iter, b_mid, est_arl);

    % if abs(est_arl - gamma) / gamma < 0.05
    %     fprintf('Converged!\n');
    %     break;
    % end

    if est_arl >= gamma && abs(est_arl - gamma) / gamma < 0.05
        fprintf('  Converged!!\n'); 
        break;
    end

    if est_arl < gamma
        b_lo = b_mid;
    else
        b_hi = b_mid;
    end

    if b_hi - b_lo < 0.05
        fprintf('Tolerance reached \n');
        break;
    end
end

b_final = b_mid;
%b_final = 100;
fprintf('Final b = %.4f  (ARL = %.1f)\n\n', b_final, est_arl);

%EDD
fprintf('EDD \n');

delays = [];
n_false = 0;
n_failure = 0;

for trial = 1:n_edd
    pre = randn(d, kappa);
    %post = sample_mixture(d, n_seq - kappa, w1, mu1, w2, mu2);
    post = sample_post(d, n_seq - kappa);
    seq = [pre, post];

    % T = run_wglr(seq, b_final, w, mu_hat, Sigma_inv);
    T = run_wglr(seq, b_final, W_set, mu_hat, Sigma_inv);

    if isinf(T)
        n_failure = n_failure + 1;
    elseif T <= kappa
        n_false = n_false + 1;
    else
        delays(end + 1) = T - kappa;
    end
end


%fprintf('Window length w     : %d\n',   w);
fprintf('Threshold b         : %.4f\n', b_final);
fprintf('Estimated ARL       : %.1f\n', est_arl);
fprintf('\n');
fprintf('Successful detections: %d / %d\n', length(delays), n_edd);
fprintf('False alarms         : %d / %d\n', n_false,        n_edd);
fprintf('Failures             : %d / %d\n', n_failure,      n_edd);
if ~isempty(delays)
    fprintf('EDD (mean)           : %.1f  (std: %.1f)\n', mean(delays), std(delays));
end
fprintf('\nPaper Table 3 Setting 1, W-GLR: EDD=30.9 (std 15.0), Success=830/1000\n');