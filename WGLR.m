%Window Limited GLR

%optimal window length w? as gamma -> inf, optimal w -> log(gamma).
%rng(42);

d = 20;
gamma = 1000;
kappa = 100;
n_seq = 1000;
n_cal = 1000;
n_edd = 1000;
w = 80;
M = 2500;

w1 = 7/8; mu1 = (1/4)*ones(d, 1);
w2 = 1/8; mu2 = zeros(d, 1);

ref_data = randn(d, M);
mu_hat = mean(ref_data, 2);
diff_ref = ref_data - mu_hat;
Sigma_hat = (diff_ref * diff_ref') / (M - 1);
Sigma_inv = inv(Sigma_hat);

fprintf('Reference estimates:\n');
fprintf('  ||mu_hat||    = %.4f  (should be ~0)\n', norm(mu_hat));
fprintf('  ||Sigma_hat - I|| = %.4f  (should be ~0)\n', norm(Sigma_hat - eye(d), 'fro'));

function S = wglr_statistic(seq, t, w, mu_hat, Sigma_inv)
S = -Inf;
kappa_lo = max(1, t - w);
kappa_hi = t - 1;

Y_centered = seq - mu_hat;
cumsum_Y = cumsum(Y_centered, 2);

for kap = kappa_lo:kappa_hi
    n_win = t - kap;

    if kap >= 1
        window_sum = cumsum_Y(:, t) - cumsum_Y(:, kap);
    else
        window_sum = sumsum_Y(:, t);
    end

    stat = (window_sum' * Sigma_inv * window_sum) / n_win;
    if stat > S
        S = stat;
    end
end
end

function T = run_wglr(seq, b, w, mu_hat, Sigma_inv)
n = size(seq, 2);
T = Inf;
for t = 1:n
    S = wglr_statistic(seq(:, 1:t), t, w, mu_hat, Sigma_inv);
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

% fprintf('Threshold\n');
% n_h0 = 10 * gamma;
% b_lo = 0.5;
% b_hi = 100.0;
% 
% for iter = 1:20
%     b_mid = (b_lo + b_hi) / 2;
%     stop_times = zeros(1, n_cal);
%     for trial = 1:n_cal
%         seq_h0 = randn(d, n_h0);
%         stop_times(trial) = run_wglr(seq_h0, b_mid, w, mu_hat, Sigma_inv);
%     end
%     stop_times(isinf(stop_times)) = n_h0;
%     est_arl = mean(stop_times);
% 
%     fprintf('Iter %2d: b = %.4f, ARL = %.1f\n', iter, b_mid, est_arl);
% 
%     if abs(est_arl - gamma) / gamma < 0.05
%         fprintf('Converged!\n');
%         break;
%     end
% 
%     if est_arl < gamma
%         b_lo = b_mid;
%     else
%         b_hi = b_mid;
%     end
% 
%     if b_hi - b_lo < 0.05
%         fprintf('Tolerance reached \n');
%         break;
%     end
% end
% 
% b_final = b_mid;
b_final = 52;
fprintf('Final b = %.4f  (ARL = %.1f)\n\n', b_final, est_arl);

%EDD
fprintf('EDD \n');

delays = []'
n_false = 0;
n_failure = 0;

for trial = 1:n_edd
    pre = randn(d, kappa);
    post = sample_mixture(d, n_seq - kappa, w1, mu1, w2, mu2);
    seq = [pre, post];

    T = run_wglr(seq, b_final, w, mu_hat, Sigma_inv);

    if isinf(T)
        n_failure = n_failure + 1;
    elseif T <= kappa
        n_false = n_false + 1;
    else
        delays(end + 1) = T - kappa;
    end
end


fprintf('Window length w     : %d\n',   w);
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