

rng(42);

d          = 20;
kappa      = 100;
n_seq      = 1000;
n_edd      = 300;    % increase to 1000 for publication quality
M          = 2500;
N          = 15;
w          = 50;     % window for proposed; block size B for Scan B
sigma_post = 2;      % the sigma labelled on Fig 7

% ---- b grid and implied ARL (analytic approx, eq 12) ----
b_vals   = [3.9, 4.0, 4.1, 4.2, 4.4, 4.6, 4.8, 5.0];
arl_vals = sqrt(2*pi) ./ b_vals .* exp(b_vals.^2 / 2);
log_arls = log10(arl_vals);
n_b      = length(b_vals);

fprintf('b grid and implied ARL:\n');
for i = 1:n_b
    fprintf('  b=%.1f  =>  ARL=%.0f  (log10=%.2f)\n', ...
            b_vals(i), arl_vals(i), log_arls(i));
end

% ---- shared reference data & bandwidth ----
ref_data = randn(d, M);

n_pairs = 500;
dists   = zeros(n_pairs,1);
for i = 1:n_pairs
    idx      = randi(M,1,2);
    dists(i) = norm(ref_data(:,idx(1)) - ref_data(:,idx(2)))^2;
end
sigma2 = median(dists);
sigma  = sqrt(sigma2);   % Scan B uses sigma (not sigma2) — kept per ScanB_paper.m
fprintf('\nKernel: sigma^2=%.4f  sigma=%.4f\n\n', sigma2, sigma);

% ---- build reference blocks (same blocks reused by both methods) ----
% Proposed uses blocks of size w; Scan B also uses block size w=B
idx_all  = randperm(M, N*w);
X_blocks = cell(N,1);
for nn = 1:N
    cols         = idx_all((nn-1)*w+1 : nn*w);
    X_blocks{nn} = ref_data(:, cols);
end

% ---- pre-compute method-specific constants ----
rho    = estimate_rho(ref_data, N, sigma2, M);
var_ZB = estimate_scanB_variance(ref_data, X_blocks, N, w, sigma);
fprintf('rho = %.4f\n', rho);
fprintf('var_ZB (Scan B) = %.6f\n\n', var_ZB);

% ---- EDD storage: (subplot, b_idx) ----
EDD_prop  = nan(3, n_b);
EDD_scanB = nan(3, n_b);

subplot_names = {'Gaussian mixture', 'Laplace', 'Uniform'};

% ---- main loops ----
for sp = 1:3
    fprintf('=== Subplot %d: %s ===\n', sp, subplot_names{sp});

    for bi = 1:n_b
        b = b_vals(bi);

        % --- Proposed ---
        del_p = [];
        for trial = 1:n_edd
            pre  = randn(d, kappa);
            post = sample_post(sp, d, n_seq-kappa, sigma_post);
            T    = run_okcusum([pre, post], b, X_blocks, N, w, sigma2, rho);
            if ~isinf(T) && T > kappa
                del_p(end+1) = T - kappa; %#ok<AGROW>
            end
        end
        if ~isempty(del_p)
            EDD_prop(sp, bi) = mean(del_p);
        end

        % --- Scan B ---
        del_s = [];
        for trial = 1:n_edd
            pre  = randn(d, kappa);
            post = sample_post(sp, d, n_seq-kappa, sigma_post);
            T    = run_scanB([pre, post], b, X_blocks, N, w, sigma, var_ZB);
            if ~isinf(T) && T > kappa
                del_s(end+1) = T - kappa; %#ok<AGROW>
            end
        end
        if ~isempty(del_s)
            EDD_scanB(sp, bi) = mean(del_s);
        end

        fprintf('  b=%.1f  EDD_prop=%.1f (%d)  EDD_scanB=%.1f (%d)\n', ...
                b, EDD_prop(sp,bi), length(del_p), ...
                   EDD_scanB(sp,bi), length(del_s));
    end
end

% ---- plot ----
col_prop  = [0.85, 0.15, 0.15];   % red  — matches paper
col_scanB = [0.95, 0.55, 0.10];   % orange — matches paper

titles = { ...
    sprintf('$q=\\{\\mathcal{N}(\\mathbf{0},I)$ w.p. 0.3, $\\mathcal{N}(\\mathbf{0},\\sigma^2 I)$ w.p. 0.7$\\}$'), ...
    sprintf('$q=\\mathrm{Laplace}(\\mathbf{0},\\sigma\\mathbf{1})$'), ...
    sprintf('$q=\\mathrm{U}[-\\sigma\\mathbf{1},\\,\\sigma\\mathbf{1}]$')};

figure('Name','Figure 7 — Proposed vs Scan B','Position',[100 100 1200 380]);
for sp = 1:3
    subplot(1,3,sp); hold on; box on; grid on;

    vp = ~isnan(EDD_prop(sp,:));
    vs = ~isnan(EDD_scanB(sp,:));

    plot(log_arls(vp), EDD_prop(sp,vp),  '-o', ...
         'Color', col_prop,  'LineWidth', 2, 'MarkerSize', 6, ...
         'DisplayName', 'Proposed');
    plot(log_arls(vs), EDD_scanB(sp,vs), '-o', ...
         'Color', col_scanB, 'LineWidth', 2, 'MarkerSize', 6, ...
         'DisplayName', 'Scan B');

    xlabel('log_{10}(ARL)', 'FontSize', 11);
    if sp == 1; ylabel('EDD', 'FontSize', 11); end
    title(titles{sp}, 'Interpreter', 'latex', 'FontSize', 9);
    legend('Location', 'northwest', 'FontSize', 10);
end
sgtitle(sprintf('Figure 7: EDD vs log-ARL  (Proposed vs Scan B,  \\sigma=%d)', ...
        sigma_post), 'FontSize', 13);

% =========================================================================
%  POST-CHANGE SAMPLERS
% =========================================================================

function X = sample_post(subplot_id, d, n, sigma)
    switch subplot_id
        case 1  % Gaussian mixture: 0.3*N(0,I) + 0.7*N(0,sigma^2*I)
            X = zeros(d, n);
            for i = 1:n
                if rand < 0.3
                    X(:,i) = randn(d,1);
                else
                    X(:,i) = sigma * randn(d,1);
                end
            end
        case 2  % Laplace(0, sigma) via inverse-CDF
            U = rand(d,n) - 0.5;
            X = -sigma * sign(U) .* log(1 - 2*abs(U));
        case 3  % Uniform[-sigma, sigma]
            X = -sigma + 2*sigma*rand(d,n);
    end
end

% =========================================================================
%  HELPERS — Proposed  (identical to OnlineKernelCusum.m)
% =========================================================================

function K = kernel_matrix(A, B, sigma2)
    K = exp(-pdist2(A', B').^2 / (2*sigma2));
end

function val = k_func(x, y, sigma2)
    dv  = x - y;
    val = exp(-(dv'*dv) / (2*sigma2));
end

function val = h_func(x1, x2, y1, y2, sigma2)
    val = k_func(x1,x2,sigma2) + k_func(y1,y2,sigma2) ...
        - k_func(x1,y2,sigma2) - k_func(x2,y1,sigma2);
end

function rho = estimate_rho(ref_data, N, sigma2, M)
    n_mc = 500;
    h2v  = zeros(n_mc,1);
    for i = 1:n_mc
        idx    = randi(M,1,4);
        hv     = h_func(ref_data(:,idx(1)), ref_data(:,idx(2)), ...
                        ref_data(:,idx(3)), ref_data(:,idx(4)), sigma2);
        h2v(i) = hv^2;
    end
    E_h2 = mean(h2v);
    cv   = zeros(n_mc,1);
    for i = 1:n_mc
        idx   = randi(M,1,6);
        cv(i) = h_func(ref_data(:,idx(1)), ref_data(:,idx(2)), ...
                       ref_data(:,idx(5)), ref_data(:,idx(6)), sigma2) * ...
                h_func(ref_data(:,idx(3)), ref_data(:,idx(4)), ...
                       ref_data(:,idx(5)), ref_data(:,idx(6)), sigma2);
    end
    Cov_h = mean(cv);
    rho   = 0.5 * (E_h2/N + (N-1)/N * Cov_h)^(-0.5);
end

function T = run_okcusum(seq, b, X_blocks, N, w, sigma2, rho)
    n_obs = size(seq,2);
    dv    = size(seq,1);
    T     = Inf;

    Y    = zeros(dv,w);
    G_XX = cell(N,1);
    for nn = 1:N
        G_XX{nn} = kernel_matrix(X_blocks{nn}, X_blocks{nn}, sigma2);
    end
    G_XY = cell(N,1);
    for nn = 1:N; G_XY{nn} = zeros(w,w); end
    G_YY = zeros(w,w);

    for t = 1:n_obs
        Yt = seq(:,t);

        Y(:,1:w-1) = Y(:,2:w);
        Y(:,w)     = Yt;

        G_YY(1:w-1,1:w-1) = G_YY(2:w,2:w);
        nc        = kernel_matrix(Y, Yt, sigma2);
        G_YY(:,w) = nc;
        G_YY(w,:) = nc';

        for nn = 1:N
            G_XY{nn}(:,1:w-1) = G_XY{nn}(:,2:w);
            G_XY{nn}(:,w)     = kernel_matrix(X_blocks{nn}, Yt, sigma2);
        end

        z = 0;
        for nn = 1:N
            z = z + G_XX{nn}(w-1,w) + G_YY(w-1,w) ...
                  - G_XY{nn}(w-1,w) - G_XY{nn}(w,w-1);
        end

        if t < w; continue; end

        Zt = 2*rho/N * z;
        for B = 3:min(w,t)
            Bt = w - B + 1;
            for nn = 1:N
                for i = Bt+1:w
                    z = z + G_XX{nn}(Bt,i) + G_YY(Bt,i) ...
                          - G_XY{nn}(Bt,i) - G_XY{nn}(i,Bt);
                end
            end
            Zt_B = (2*sqrt(2)*rho) / (N*sqrt(B*(B-1))) * z;
            Zt   = max(Zt, Zt_B);
        end

        if Zt > b; T = t; return; end
    end
end

% =========================================================================
%  HELPERS — Scan B  (identical to ScanB_paper.m)
%  Note: Scan B was written with sigma (not sigma2) — kept exactly as-is
% =========================================================================

function var_out = estimate_scanB_variance(ref_data, ref_blocks, N, B, sigma)
    % sigma here is sqrt(sigma2), matching ScanB_paper.m convention
    n_mc  = 500;
    M_ref = size(ref_data,2);
    h2v   = zeros(n_mc,1);
    for i = 1:n_mc
        idx    = randi(M_ref,1,4);
        hv     = h_func_s(ref_data(:,idx(1)), ref_data(:,idx(2)), ...
                          ref_data(:,idx(3)), ref_data(:,idx(4)), sigma);
        h2v(i) = hv^2;
    end
    E_h2 = mean(h2v);
    cv   = zeros(n_mc,1);
    for i = 1:n_mc
        idx   = randi(M_ref,1,6);
        cv(i) = h_func_s(ref_data(:,idx(1)), ref_data(:,idx(2)), ...
                         ref_data(:,idx(5)), ref_data(:,idx(6)), sigma) * ...
                h_func_s(ref_data(:,idx(3)), ref_data(:,idx(4)), ...
                         ref_data(:,idx(5)), ref_data(:,idx(6)), sigma);
    end
    Cov_h   = mean(cv) - mean(cv).^2;   % matches ScanB_paper.m line exactly
    C       = 2 / (B*(B-1));
    var_out = C * (E_h2/N + (N-1)/N * Cov_h);
end

function val = h_func_s(x1, x2, y1, y2, sigma)
    % sigma-based version — matches ScanB_paper.m
    val = k_func_s(x1,x2,sigma) + k_func_s(y1,y2,sigma) ...
        - k_func_s(x1,y2,sigma) - k_func_s(x2,y1,sigma);
end

function val = k_func_s(x1, x2, sigma)
    val = exp(-norm(x1-x2)^2 / (2*sigma^2));
end

function K = kernel_matrix_s(A, B, sigma)
    K = exp(-pdist2(A', B').^2 / (2*sigma^2));
end

function mmd_u = MMD_u(X, Y, sigma)
    K_XX = kernel_matrix_s(X, X, sigma);
    K_YY = kernel_matrix_s(Y, Y, sigma);
    K_XY = kernel_matrix_s(X, Y, sigma);
    B0   = size(X,2);
    H    = K_XX + K_YY - K_XY - K_XY';
    H    = H - diag(diag(H));
    mmd_u = sum(H(:)) / (B0*(B0-1));
end

function D_hat = compute_D_hat(ref_blocks, Y, N, sigma)
    vals = zeros(N,1);
    for nn = 1:N
        vals(nn) = MMD_u(ref_blocks{nn}, Y, sigma);
    end
    D_hat = mean(vals);
end

function T = run_scanB(seq, b, ref_blocks, N, B, sigma, var_ZB)
    % Identical to ScanB_paper.m
    n      = size(seq,2);
    T      = Inf;
    std_ZB = sqrt(var_ZB);
    for t = B:n
        Y     = seq(:, t-B+1:t);
        D_hat = compute_D_hat(ref_blocks, Y, N, sigma);
        ZB    = D_hat / std_ZB;
        if ZB > b; T = t; return; end
    end
end
