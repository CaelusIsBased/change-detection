% Window Limited CUSUM

%rng(42);

d = 20;
gamma = 1000;
kappa = 100;
n_seq = 1000;
n_cal = 1000;
n_edd = 1000;
%w = 4;
w = 80;
%w = 10;

%d = 20;
% w1 = 7/8; mu1 = (1/4)*ones(d, 1);
% w2 = 1/8; mu2 = zeros(d, 1);

function lp = log_p(Y, d)
lp = -0.5 * sum(Y.^2, 1) - (d/2)*log(2*pi);
end

function lq = log_q_estimated(y, mu_hat, d)
    y_c = y - mu_hat;
    lq  = -0.5 * sum(y_c.^2) - (d/2)*log(2*pi);
end

function T = run_wcusum(seq, b, w, d)
n = size(seq, 2);
S = 0;
T = Inf;

for t = w+1:n          % <-- start at w+1, not 1
    window = seq(:, t-w:t-1);    % exactly w samples, not including Y_t
    mu_hat = mean(window, 2);

    lq  = -0.5 * sum((seq(:,t) - mu_hat).^2) - (d/2)*log(2*pi);
    lp  = -0.5 * sum(seq(:,t).^2) - (d/2)*log(2*pi);
    llr = lq - lp;

    S = max(0, S + llr);
    if S > b
        T = t;
        return;
    end
end
end

% For case 1 and 2
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



% For case 3, 4, 5
% post = sample_post(d, n_seq - kappa);
 
%various sample mixtures to try
%case2
% d  = 50;
% w1 = 1/2; mu1 = (1/3)*ones(d,1);
% w2 = 1/2; mu2 = zeros(d,1);

test_post = sample_mixture(d, n, w1, mu1, w2, mu2);  % or however you call it
fprintf('Mean of post-change: %.4f\n', mean(mean(test_post)));
fprintf('Std of post-change: %.4f\n', mean(std(test_post)));

%case3
% sampling only — LLR unchanged
% d = 20;
% function X = sample_post(d, n)
%     U = rand(d, n) - 0.5;
%     X = 0.5 - 0.25 * sign(U) .* log(1 - 2*abs(U));
% end

%case4
% d = 20;
% function X = sample_post(d, n)
%     X = -1 + (4/5) * (-log(rand(d,n)));
% end

%case5
d = 20;
function X = sample_post(d, n)
    X = -0.5 + 2*rand(d,n);
end

%no need to change llr

fprintf("Threshold \n");
n_h0 = 10* gamma;
b_lo = 0.5;
b_hi = 30.0;

for iter = 1:20
    b_mid = (b_lo + b_hi) / 2;
    stop_times = zeros(1, n_cal);
    for trial = 1:n_cal
        seq_h0 = randn(d, n_h0);
        stop_times(trial) = run_wcusum(seq_h0, b_mid, w, d);
    end
    stop_times(isinf(stop_times)) = n_h0;
    est_arl = mean(stop_times);

    fprintf('Iter %2d: b = %.4f, ARL = %.1f\n', iter, b_mid, est_arl);

    if abs(est_arl - gamma) / gamma < 0.05
        fprintf('Converged!\n');
        break;
    end
    if est_arl < gamma
        b_lo = b_mid;
    else
        b_hi = b_mid;
    end

    if b_hi - b_lo < 0.05
        fprintf('Tolerance reached\n');
        break;
    end
end

b_final = b_mid;
fprintf('Final b = %.4f (ARL = %.1f)\n\n', b_final, est_arl);

%EDD
fprintf('EDD \n');

delays = [];
n_false = 0;
n_failure = 0;

for trial = 1:n_edd
    pre = randn(d, kappa);
    % post = sample_mixture(d, n_seq - kappa, w1, mu1, w2, mu2);
    post = sample_post(d, n_seq - kappa);
    seq = [pre, post];

    T = run_wcusum(seq, b_final, w, d);

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

%only for case1
%fprintf('\nEDD=163.3 (std 130.6), Success=870/1000\n');

