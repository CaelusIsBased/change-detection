% Online Kernel CUSUM (for the nth time)
 
rng(42);
 
%d       = 20;
gamma   = 1000;
kappa   = 100;
n_seq   = 1000;
n_cal   = 10;
n_edd   = 1000;
M       = 2500;
N       = 15;
w       = 80;

% case 1
% d = 20;
% pw1 = 7/8;  mu1 = (1/4)*ones(d,1);
% pw2 = 1/8;  mu2 = zeros(d,1);

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
dists   = zeros(n_pairs, 1);
for i = 1:n_pairs
    idx      = randi(M, 1, 2);
    dists(i) = norm(ref_data(:,idx(1)) - ref_data(:,idx(2)))^2;
end
sigma2 = median(dists);
fprintf('Kernel bandwidth sigma^2 = %.4f\n', sigma2);
idx_all  = randperm(M, N*w);
X_blocks = cell(N, 1);
for n = 1:N
    cols       = idx_all((n-1)*w+1 : n*w);
    X_blocks{n} = ref_data(:, cols);  
end
 
rho = estimate_rho(ref_data, N, sigma2, M);
fprintf('Estimated rho = %.4f\n\n', rho);
 
function K = kernel_matrix(A, B, sigma2)
    K = exp(-pdist2(A', B').^2 / (2*sigma2));
end

function rho = estimate_rho(ref_data, N, sigma2, M)
    n_mc = 500;
    h2_vals = zeros(n_mc, 1);
    for i = 1:n_mc
        idx = randi(M, 1, 4);
        x1  = ref_data(:, idx(1));
        x2  = ref_data(:, idx(2));
        y1  = ref_data(:, idx(3));
        y2  = ref_data(:, idx(4));
        hval = h_func(x1, x2, y1, y2, sigma2);
        h2_vals(i) = hval^2;
    end
    E_h2 = mean(h2_vals);
 
    cov_vals = zeros(n_mc, 1);
    for i = 1:n_mc
        idx = randi(M, 1, 6);
        x1  = ref_data(:, idx(1));
        x2  = ref_data(:, idx(2));
        x3  = ref_data(:, idx(3));
        x4  = ref_data(:, idx(4));
        y1  = ref_data(:, idx(5));
        y2  = ref_data(:, idx(6));
        cov_vals(i) = h_func(x1,x2,y1,y2,sigma2) * h_func(x3,x4,y1,y2,sigma2);
    end
   
    Cov_h = mean(cov_vals);
 
    inner = E_h2/N + (N-1)/N * Cov_h;
    rho   = 0.5 * inner^(-0.5);
end

function val = h_func(x1, x2, y1, y2, sigma2)
val = k_func(x1, x2, sigma2) + k_func(y1, y2, sigma2) ...
    - k_func(x1, y2, sigma2) - k_func(x2, y1, sigma2);
end

function val = k_func(x, y, sigma2)
diff = x - y;
val = exp(-(diff' * diff) / (2 * sigma2));
end

function T = run_okcusum(seq, b, X_blocks, N, w, sigma2, rho)
n_obs = size(seq, 2);
T = Inf;

Y = zeros(size(seq, 1), w);

G_XX = cell(N, 1);
for nn = 1:N
    G_XX{nn} = kernel_matrix(X_blocks{nn}, X_blocks{nn}, sigma2);
end

G_XY = cell(N, 1);
for nn = 1:N
    G_XY{nn} = zeros(w, w);
end

G_YY = zeros(w, w);

for t = 1:n_obs
    Yt = seq(:, t);

    Y(:, 1:w-1) = Y(:, 2:w);
    Y(:, w) = Yt;

    G_YY(1:w-1, 1:w-1) = G_YY(2:w, 2:w);

    new_col = kernel_matrix(Y, Yt, sigma2);
    G_YY(:, w) = new_col;
    G_YY(w, :) = new_col;

    for nn = 1:N
        G_XY{nn}(:, 1:w-1) = G_XY{nn}(:, 2:w);
        G_XY{nn}(:, w) = kernel_matrix(X_blocks{nn}, Yt, sigma2);
    end

    z = 0;
    for nn = 1:N
        z = z + G_XX{nn}(w-1, w) ...
            + G_YY(w - 1, w) ...
            - G_XY{nn}(w - 1, w) ...
            - G_XY{nn}(w, w-1);
    end

    if t < w
        continue
    end
    %Zt = sqrt(2) * rho / N * z;
    Zt = 2 * rho / N * z;

    B_max = min(w, t);
    for B = 3:B_max
        B_tilde = w - B + 1;

        for nn = 1:N
            for i = B_tilde + 1 : w 
                z = z + G_XX{nn}(B_tilde, i) ...
                    + G_YY(B_tilde, i) ...
                    - G_XY{nn}(B_tilde, i) ...
                    - G_XY{nn}(i, B_tilde);
            end
        end

        %Zt_B = (2*rho) / (N * sqrt(B*(B-1))) * z;
        Zt_B = (2*sqrt(2)*rho) / (N * sqrt(B*(B-1))) * z;
        Zt = max(Zt, Zt_B);
    end

    if Zt > b
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
b_lo = 0.5;
b_hi = 20;

for iter = 1:20
    b_mid = (b_lo + b_hi) / 2;

    stop_times = zeros(1, n_cal);
    for trial = 1:n_cal
        seq_h0 = randn(d, n_h0);
        stop_times(trial) = run_okcusum(seq_h0, b_mid, X_blocks, N, w, sigma2, rho);
    end
    stop_times(isinf(stop_times)) = n_h0;
    est_arl = mean(stop_times);

    fprintf('  Iter %2d: b = %.4f,  ARL = %.1f\n', iter, b_mid, est_arl);

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
fprintf('Final b = %.4f  (ARL = %.1f)\n\n', b_final, est_arl);

%EDD
fprintf("EDD\n");

delays = [];
n_false = 0;
n_failure = 0;

for trial = 1:n_edd
    pre = randn(d, kappa);
    % post = sample_mixture(d, n_seq - kappa, pw1, mu1, pw2, mu2);
    post = sample_post(d, n_seq - kappa);
    seq = [pre, post];

    T = run_okcusum(seq, b_final, X_blocks, N, w, sigma2, rho);

    if isinf(T)
        n_failure = n_failure + 1;
    elseif T <= kappa
        n_false = n_false + 1;
    else
        delays(end + 1) = T - kappa;
    end
end

fprintf('N = %d, w = %d\n',          N,       w);
fprintf('Threshold b         : %.4f\n', b_final);
fprintf('Estimated ARL       : %.1f\n', est_arl);
fprintf('\n');
fprintf('Successful detections: %d / %d\n', length(delays), n_edd);
fprintf('False alarms         : %d / %d\n', n_false,        n_edd);
fprintf('Failures             : %d / %d\n', n_failure,      n_edd);
if ~isempty(delays)
    fprintf('EDD (mean)           : %.1f  (std: %.1f)\n', mean(delays), std(delays));
end
fprintf('\nPaper Table 3 Setting 1, Proposed: EDD=28.6 (std 14.4), Success=794/1000\n');

