%Kernel CUSUM paper

rng(42);

%d = 20;
gamma = 1000;
kappa = 100;
n_seq = 1000;
n_cal = 1000;
n_edd = 1000;
M = 2500;
delta = 1/50;

% case 1
% d = 20;
% pw1 = 7/8; mu1 = (1/4) * ones(d, 1);
% pw2 = 1/8; mu2 = zeros(d, 1);

%case2
% d  = 50;
% pw1 = 1/2; mu1 = (1/3)*ones(d,1);
% pw2 = 1/2; mu2 = zeros(d,1);

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

n_pairs = 500;
dists = zeros(n_pairs, 1);
for i = 1:n_pairs
    idx = randi(M, 1, 2);
    dists(i) = norm(ref_data(:, idx(1)) - ref_data(:, idx(2)))^2;
end
sigma2 = median(dists);
fprintf('Kernel bandwidth sigma^2 = %.4f\n\n', sigma2);

function k = gauss_kernel(x, y, sigma2)
diff = x - y;
k = exp(-(diff' * diff) / (2 * sigma2));
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

function T = run_kcusum(seq, h, ref_data, M, delta, sigma2)
n = size(seq, 2);
Z = 0;
T = Inf;

ref_idx = randi(M, 1, n);
ref_stream = ref_data(:, ref_idx);

for t = 2:n
    if mod(t, 2) == 0
        x0 = seq(:, t-1);
        x1 = seq(:, t);
        y0 = ref_stream(:, t-1);
        y1 = ref_stream(:, t);

        v = gauss_kernel(x0, x1, sigma2) ...
            + gauss_kernel(y0, y1, sigma2) ...
            - gauss_kernel(x0, y1, sigma2) ...
            - gauss_kernel(x1, y0, sigma2) ...
            - delta;

    else
        v = 0;
    end

    Z = max(0, Z + v);

    if Z > h
        T = t;
        return;
    end
end
end

fprintf("Threshold \n");
n_h0 = 10 * gamma;
b_lo = 0.1;
b_hi = 20;

for iter = 1:20
    b_mid = (b_lo + b_hi) / 2;

    stop_times = zeros(1, n_cal);
    for trial = 1:n_cal
        seq_h0 = randn(d, n_h0);
        stop_times(trial) = run_kcusum(seq_h0, b_mid, ref_data, M, delta, sigma2);
    end
    stop_times(isinf(stop_times)) = n_h0;
    est_arl = mean(stop_times);

    fprintf('  Iter %2d: b = %.4f,  ARL = %.1f\n', iter, b_mid, est_arl)

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
fprintf('Final b = %.4f  (ARL = %.1f)\n\n', b_final, est_arl);

%EDD
fprintf("EDD \n");

delays = [];
n_false = 0;
n_failure = 0;

for trial = 1:n_edd
    pre = randn(d, kappa);
    %post = sample_mixture(d, n_seq - kappa, pw1, mu1, pw2, mu2);
    post = sample_post(d, n_seq - kappa);
    seq = [pre, post];

    T = run_kcusum(seq, b_final, ref_data, M, delta, sigma2);

    if isinf(T)
        n_failure = n_failure + 1;
    elseif T <= kappa
        n_false = n_false + 1;
    else
        delays(end + 1) = T - kappa;
    end
end

fprintf('delta               : %.4f\n', delta);
fprintf('sigma^2             : %.4f\n', sigma2);
fprintf('Threshold b         : %.4f\n', b_final);
fprintf('Estimated ARL       : %.1f\n', est_arl);
fprintf('\n');
fprintf('Successful detections: %d / %d\n', length(delays), n_edd);
fprintf('False alarms         : %d / %d\n', n_false,        n_edd);
fprintf('Failures             : %d / %d\n', n_failure,      n_edd);
if ~isempty(delays)
    fprintf('EDD (mean)           : %.1f  (std: %.1f)\n', mean(delays), std(delays));
end
fprintf('\nPaper Table 3 Setting 1, KCUSUM: EDD=296.3 (std 209.8), Success=607/1000\n');

