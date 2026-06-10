%Scan B for the papaer

rng(42);

d = 20;
gamma = 1000;
kappa = 100;
n_seq = 1000;
n_cal = 10;
n_edd = 1000;
M = 2500;
N = 15;
B = 80;

% case 1
% d = 20;
% pw1 = 7/8; mu1 = (1/4) * ones(d, 1);
% pw2 = 1/8; mu2 = zeros(d, 1);

%case2
d  = 50;
pw1 = 1/2; mu1 = (1/3)*ones(d,1);
pw2 = 1/2; mu2 = zeros(d,1);

%case 3
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

sigma = sqrt(median(dists));

fprintf('Estimated kernel bandwidth sigma = %.4f\n', sigma);

ref_blocks = build_ref_blocks(ref_data, N, B, M);

var_ZB = estimate_variance(ref_data, ref_blocks, N, B, sigma);
fprintf('Estimated Var(D_hat_B) = %.6f\n', var_ZB);
fprintf('Std(D_hat_B) = %.6f\n\n', sqrt(var_ZB));

function blocks = build_ref_blocks(ref_data, N, B, M)
idx = randperm(M, N*B);
blocks = cell(N, 1);
for n = 1:N
    cols = idx((n-1)*B+1 : n*B);
    blocks{n} = ref_data(:, cols);
end
end

function K = kernel_matrix(A, B_mat, sigma)
K = exp(-pdist2(A', B_mat').^2 / (2 * sigma^2));
end

function mmd_u = MMD_u(X, Y, sigma)
K_XX = kernel_matrix(X, X, sigma);
K_YY = kernel_matrix(Y, Y, sigma);
K_XY = kernel_matrix(X, Y, sigma);

B0 = size(X, 2);
H = K_XX + K_YY - K_XY - K_XY';
H = H - diag(diag(H));
mmd_u = sum(H(:)) / (B0 * (B0 - 1));
end

function D_hat = compute_D_hat(ref_blocks, Y, N, sigma)
mmd_vals = zeros(N, 1);
for n = 1:N
    mmd_vals(n) = MMD_u(ref_blocks{n}, Y, sigma);
end
D_hat = mean(mmd_vals);
end

function var_out = estimate_variance(ref_data, ref_blocks, N, B, sigma)
n_mc = 500;
h2_vals = zeros(n_mc, 1);
M_ref   = size(ref_data, 2);
for i = 1:n_mc
    idx  = randi(M_ref, 1, 4);
    x1   = ref_data(:, idx(1));
    x2   = ref_data(:, idx(2));
    y1   = ref_data(:, idx(3));
    y2   = ref_data(:, idx(4));
    hval = h_func(x1, x2, y1, y2, sigma);
    h2_vals(i) = hval^2;
end
E_h2 = mean(h2_vals);

cov_vals = zeros(n_mc, 1);
for i = 1:n_mc
    idx  = randi(M_ref, 1, 6);
    x1   = ref_data(:, idx(1));
    x2   = ref_data(:, idx(2));
    x3   = ref_data(:, idx(3));
    x4   = ref_data(:, idx(4));
    y1   = ref_data(:, idx(5));
    y2   = ref_data(:, idx(6));
    cov_vals(i) = h_func(x1, x2, y1, y2, sigma) * ...
                  h_func(x3, x4, y1, y2, sigma);
end
Cov_h = mean(cov_vals) - mean(cov_vals).^2;  

C       = 2 / (B * (B - 1));
var_out = C * (E_h2/N + (N-1)/N * Cov_h);
end
 
function val = h_func(x1, x2, y1, y2, sigma)
    val = k_func(x1,x2,sigma) + k_func(y1,y2,sigma) ...
        - k_func(x1,y2,sigma) - k_func(x2,y1,sigma);
end
 
function val = k_func(x1, x2, sigma)
    val = exp(-norm(x1-x2)^2 / (2*sigma^2));
end
 
function T = run_scanB(seq, b, ref_blocks, N, B, sigma, var_ZB)
    n   = size(seq, 2);
    T   = Inf;
    std_ZB = sqrt(var_ZB);
 
    for t = B:n
        Y     = seq(:, t-B+1:t);   
        D_hat = compute_D_hat(ref_blocks, Y, N, sigma);
        ZB    = D_hat / std_ZB;
 
        if ZB > b
            T = t;
            return;
        end
    end
end
 
function X = sample_mixture(d, n, pw1, mu1, pw2, mu2)
    X = zeros(d, n);
    for i = 1:n
        if rand < pw1
            X(:,i) = mu1 + randn(d,1);
        else
            X(:,i) = mu2 + randn(d,1);
        end
    end
end
 
fprintf('Threshold Calibration\n');
fprintf('N = %d, B = %d, Target ARL = %d\n', N, B, gamma);
 
n_h0 = 10 * gamma;
b_lo = 0.5;
b_hi = 20.0;
 
for iter = 1:20
    b_mid = (b_lo + b_hi) / 2;
 
    stop_times = zeros(1, n_cal);
    for trial = 1:n_cal
        seq_h0              = randn(d, n_h0);
        stop_times(trial)   = run_scanB(seq_h0, b_mid, ref_blocks, N, B, sigma, var_ZB);
    end
    stop_times(isinf(stop_times)) = n_h0;
    est_arl = mean(stop_times);
 
    fprintf('  Iter %2d: b = %.4f,  ARL = %.1f\n', iter, b_mid, est_arl);
 
    if abs(est_arl - gamma) / gamma < 0.05
        fprintf('  Converged.\n');
        break;
    end
    if est_arl < gamma
        b_lo = b_mid;
    else
        b_hi = b_mid;
    end
    if b_hi - b_lo < 0.05
        fprintf('  Tolerance reached.\n');
        break;
    end
end
 
b_final = b_mid;
fprintf('Final b = %.4f  (ARL = %.1f)\n\n', b_final, est_arl);
 
%EDD
fprintf('EDD\n');
 
delays    = [];
n_false   = 0;
n_failure = 0;
 
for trial = 1:n_edd
    pre  = randn(d, kappa);
    %post = sample_mixture(d, n_seq-kappa, pw1, mu1, pw2, mu2);
    post = sample_post(d, n_seq - kappa);
    seq  = [pre, post];
 
    T = run_scanB(seq, b_final, ref_blocks, N, B, sigma, var_ZB);
 
    if isinf(T)
        n_failure = n_failure + 1;
    elseif T <= kappa
        n_false = n_false + 1;
    else
        delays(end+1) = T - kappa;
    end
end


fprintf('N = %d, B = %d\n',          N,            B);
fprintf('Threshold b         : %.4f\n', b_final);
fprintf('Estimated ARL       : %.1f\n', est_arl);
fprintf('\n');
fprintf('Successful detections: %d / %d\n', length(delays), n_edd);
fprintf('False alarms         : %d / %d\n', n_false,        n_edd);
fprintf('Failures             : %d / %d\n', n_failure,      n_edd);
if ~isempty(delays)
    fprintf('EDD (mean)           : %.1f  (std: %.1f)\n', mean(delays), std(delays));
end
fprintf('\nPaper Table 3 Setting 1, Scan B: EDD=35.4 (std 10.5), Success=794/1000\n');