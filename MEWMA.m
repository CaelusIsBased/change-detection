%MEWMA (MEWWW)

%rng(42);

d = 20;
gamma = 1000;
kappa = 100;
n_seq = 1000;
n_cal = 1000;
n_edd = 1000;
M = 2500;
r = 0.3; %tune!!!

pw1 = 7/8; mu1 = (1/4)*ones(d, 1);
pw2 = 1/8; mu2 = zeros(d, 1);

ref_data = randn(d, M);
mu_ref = mean(ref_data, 2);
diff_ref = ref_data - mu_ref;
Sigma_0 = (diff_ref * diff_ref')/(M - 1);

fprintf('||Sigma_0 - I|| = %.4f (should be ~0)\n\n', norm(Sigma_0 - eye(d), 'fro'));

function T = run_mewma(seq, b, r, Sigma_0, d)
n = size(seq, 2);
S = zeros(d, 1);
T = Inf;

for t = 1:n
    S = r*seq(:, t) + (1-r) * S;
    Sigma_t = (r/(2-r)) * (1 - (1-r)^(2*t)) * Sigma_0;

    W = S' * (Sigma_t \ S);

    if W > b
        T = t;
        return;
    end
end
end

function X = sample_mixture(d, n, pw1, mu1, pw2, mu2)
X = zeros(d, n);
for i = 1:n
    if rand < pw1
        X(:, i) = mu1 + randn(d, 1);
    else
        X(:, i) = mu2 + randn(d, 1);
    end
end
end

fprintf("Threshold\n");

n_h0 = 10 * gamma;
b_lo = 1;0;
b_hi = 100;

for iter = 1:20
    b_mid = (b_lo + b_hi) / 2;

    stop_times = zeros(1, n_cal);
    for trial = 1:n_cal
        seq_h0 = randn(d, n_h0);
        stop_times(trial) = run_mewma(seq_h0, b_mid, r, Sigma_0, d);
    end

    stop_times(isinf(stop_times)) = n_h0;
    est_arl = mean(stop_times);

    fprintf('  Iter %2d: b = %.4f,  ARL = %.1f\n', iter, b_mid, est_arl);

    if abs(est_arl - gamma) / gamma < 0.05
        fprintf('Converged! \n');
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
fprintf('Final b = %.4f (ARL = %.1f)\n\n', b_final, est_arl);

% EDD

fprintf("EDD\n");

delays = [];
n_false = 0;
n_failure = 0;

for trial = 1:n_edd
    pre = randn(d, kappa);
    post = sample_mixture(d, n_seq - kappa, pw1, mu1, pw2, mu2);
    seq = [pre, post];

    T = run_mewma(seq, b_final, r, Sigma_0, d);

    if isinf(T)
        n_failure = n_failure + 1;
    elseif T <= kappa
        n_false = n_false + 1;
    else
        delays(end + 1) = T - kappa;
    end
end


fprintf('r_decay             : %.2f\n',   r);
fprintf('Threshold b         : %.4f\n',   b_final);
fprintf('Estimated ARL       : %.1f\n',   est_arl);
fprintf('\n');
fprintf('Successful detections: %d / %d\n', length(delays), n_edd);
fprintf('False alarms         : %d / %d\n', n_false,        n_edd);
fprintf('Failures             : %d / %d\n', n_failure,      n_edd);
if ~isempty(delays)
    fprintf('EDD (mean)           : %.1f  (std: %.1f)\n', mean(delays), std(delays));
end
fprintf('\nPaper Table 3 Setting 1, MEWMA: EDD=83.6 (std 77.4), Success=815/1000\n');