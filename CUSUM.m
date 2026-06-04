% Classic CUSUM
%rng(42);

d = 20;
gamma = 1000;
kappa = 100;
n_seq = 1000;
n_cal = 1000;
n_edd = 1000;

w1 = 7/8;  mu1 = (1/4) * ones(d,1);  Sigma1 = eye(d);
w2 = 1/8;  mu2 = zeros(d,1);         Sigma2 = eye(d);

function llr = log_likelihood_ratio(Y, d, w1, mu1, Sigma1, w2, mu2, Sigma2)
log_p = -0.5 * sum(Y.^2, 1) - (d/2) * log(2*pi);

Y_c1 = Y - mu1;
Y_c2 = Y - mu2;

log_q1 = -0.5 * sum(Y_c1.^2, 1) - (d/2)*log(2*pi);
log_q2 = -0.5 * sum(Y_c2.^2, 1) - (d/2)*log(2*pi);

log_w1_q1 = log(w1) + log_q1;
log_w2_q2 = log(w2) + log_q2;

max_log = max(log_w1_q1, log_w2_q2);
log_q = max_log + log(exp(log_w1_q1 - max_log) + exp(log_w2_q2 - max_log));

llr = log_q - log_p;
end

%setting 3
% function llr = log_likelihood_ratio(Y, d)
%     % p = N(0, I),  q = Lap(1/2, 1/4) per coordinate
%     log_p = -0.5 * sum(Y.^2, 1) - (d/2)*log(2*pi);
%     % Laplace: log f(y; mu, b) = -log(2b) - |y-mu|/b
%     log_q = sum(-log(2 * 0.25) - abs(Y - 0.5) / 0.25, 1);
%     llr = log_q - log_p;
% end

%setting4
% function llr = log_likelihood_ratio(Y, d)
%     % p = N(0, I),  q = Exp(-1, 4/5) per coordinate
%     % Shifted exponential: f(y; mu, lambda) = (1/scale)*exp(-(y-mu)/scale) for y >= mu
%     % Here mu = -1, scale = 4/5
%     log_p = -0.5 * sum(Y.^2, 1) - (d/2)*log(2*pi);
%     mu_exp = -1;  scale = 4/5;
%     % Check support: all coordinates must be >= mu
%     valid = all(Y >= mu_exp, 1);
%     log_q = sum(-log(scale) - (Y - mu_exp)/scale, 1);
%     log_q(~valid) = -Inf;
%     llr = log_q - log_p;
% end

%setting5
% function llr = log_likelihood_ratio(Y, d)
%     % p = N(0, I),  q = Exp(-1, 4/5) per coordinate
%     % Shifted exponential: f(y; mu, lambda) = (1/scale)*exp(-(y-mu)/scale) for y >= mu
%     % Here mu = -1, scale = 4/5
%     log_p = -0.5 * sum(Y.^2, 1) - (d/2)*log(2*pi);
%     mu_exp = -1;  scale = 4/5;
%     % Check support: all coordinates must be >= mu
%     valid = all(Y >= mu_exp, 1);
%     log_q = sum(-log(scale) - (Y - mu_exp)/scale, 1);
%     log_q(~valid) = -Inf;
%     llr = log_q - log_p;
% end

% For Settings 3-5 I dropped the distribution parameters from the function 
% signature since they're hardcoded — just swap the whole function out when 
% switching settings.
% Setting 4's support check is important — a Gaussian sample will sometimes 
% fall below -1, giving -Inf LLR, which max(0, S + llr) handles correctly 
% by resetting to 0.
% For Setting 2 just change d=50 and the mixture parameters, 
% the function itself is identical to Setting 1.


function T = run_cusum(seq, b, d, w1, mu1, Sigma1, w2, mu2, Sigma2)
n = size(seq, 2);
S = 0;
T = Inf;
for t = 1:n
    llr = log_likelihood_ratio(seq(:,t), d, w1, mu1, Sigma1, w2, mu2, Sigma2);
    S = max(0, S + llr);
    if S > b
        T = t;
        return;
    end
end
end

fprintf("Step1: Calibrate Threshold\n");
fprintf("Target ARL = %d\n", gamma);

n_h0 = 10 * gamma; %???????

b_lo = 0.5;
b_hi = 30;
tol = 0.05;

for iter = 1:20
    b_mid = (b_lo + b_hi) / 2;
    stop_times = zeros(1, n_cal);
    for trial = 1:n_cal
        seq_h0 = randn(d, n_h0);
        stop_times(trial) = run_cusum(seq_h0, b_mid, d, w1, mu1, Sigma1, w2, mu2, Sigma2);
    end
    stop_times(isinf(stop_times)) = n_h0;
    est_arl = mean(stop_times);

    fprintf('Iter %2d: b = %.4f, estimated ARL = %.1f\n', iter, b_mid, est_arl);
    if abs(est_arl - gamma) / gamma < 0.05
        fprintf('Converged!\n');
        break;
    end

    if est_arl < gamma
        b_lo = b_mid;
    else
        b_hi = b_mid;
    end

    if b_hi - b_lo < tol
        fprintf('Binaryh tol reached\n');
        break;
    end
end

b_final = b_mid;

fprintf('Final Threhsold b = %.4f (estimated arl = %.1f)\n\n', b_final, est_arl);

fprintf('Step2: EDD Eval\n');

delays = [];
n_false = 0;
n_failure = 0;

for trial = 1:n_edd
    pre = randn(d, kappa);
    n_post = n_seq - kappa;
    post = sample_mixture(d, n_post, w1, mu1, Sigma1, w2, mu2, Sigma2);

    seq = [pre, post];

    T = run_cusum(seq, b_final, d, w1, mu1, Sigma1, w2, mu2, Sigma2);

    if isinf(T)
        n_failure = n_failure + 1;
    elseif T <= kappa
        n_false = n_false + 1;
    else
        delays(end+1) = T - kappa;
    end
end

function X = sample_mixture(d, n, w1, mu1, Sigma1, w2, mu2, Sigma2)
X = zeros(d, n);
for i = 1:n
    if rand < w1
        X(:, i) = mu1 + chol(Sigma1)' * randn(d, 1);
    else
        X(:, i) = mu2 + chol(Sigma2)' * randn(d, 1);
    end
end
end

n_success = length(delays);

fprintf('\n========== RESULTS ==========\n');
fprintf('Threshold b         : %.4f\n',   b_final);
fprintf('Estimated ARL       : %.1f\n',   est_arl);
fprintf('Target ARL (gamma)  : %d\n',     gamma);
fprintf('-----------------------------\n');
fprintf('Successful detections: %d / %d\n', n_success, n_edd);
fprintf('False alarms         : %d / %d\n', n_false,   n_edd);
fprintf('Failures             : %d / %d\n', n_failure, n_edd);

if n_success > 0
    edd = mean(delays);
    edd_std = std(delays);
    fprintf('EDD (mean delay): %.1f (std: %.1f)\n', edd, edd_std);
else
    fprintf('EDD: N/A\n');
end
fprintf(' EDD = 13.3 (std 6.5), Success = 840/1000\n');