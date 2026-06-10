% Fig. 2 — Trajectory plot
% Pre-change: N(0, I_20)
% Post-change: 0.3*N(0, I_20) + 0.7*N(0, 4*I_20)
% Change-point at tau = 51, plot t = 1..125, 100 trials

rng(42);

d       = 20;
M       = 2500;
N       = 15;
w       = 50;   % window size — Fig.2 uses w=50 (paper p.8: "platform at w=50 steps")
B_scanb = 50;   % Scan-B block size = w (same window, reacts from t=w=50 onward)

n_trials = 100;
tau      = 51;
T_max    = 125;

% ── Reference data & kernel bandwidth ────────────────────────────────────
ref_data = randn(d, M);

n_pairs = 500;
dists   = zeros(n_pairs, 1);
for i = 1:n_pairs
    idx      = randi(M, 1, 2);
    dists(i) = norm(ref_data(:,idx(1)) - ref_data(:,idx(2)))^2;
end
sigma2 = median(dists);
sigma  = sqrt(sigma2);
fprintf('sigma^2 = %.4f,  sigma = %.4f\n', sigma2, sigma);

% ── Reference blocks (shared by both methods) ────────────────────────────
idx_all  = randperm(M, N*w);
X_blocks = cell(N, 1);
for n = 1:N
    cols        = idx_all((n-1)*w+1 : n*w);
    X_blocks{n} = ref_data(:, cols);
end

% ── Proposed: estimate rho ───────────────────────────────────────────────
rho = estimate_rho(ref_data, N, sigma2, M);
fprintf('rho = %.6f\n', rho);

% ── Scan-B: estimate variance of D_hat ───────────────────────────────────
ref_blocks_scanb = build_ref_blocks(ref_data, N, B_scanb, M);
var_ZB   = estimate_variance_scanb(ref_data, ref_blocks_scanb, N, B_scanb, sigma);
std_ZB   = sqrt(var_ZB);
fprintf('std(ZB) = %.6f\n', std_ZB);
% These should satisfy: std_ZB ≈ sqrt(2/(B*(B-1))) / (2*rho)
implied_rho = sqrt(2/(B_scanb*(B_scanb-1))) / (2*std_ZB);
fprintf('implied rho from std_ZB = %.6f  (should ≈ rho above)\n\n', implied_rho);

% ── Collect trajectories ─────────────────────────────────────────────────
traj_proposed = zeros(n_trials, T_max);
traj_scanb    = zeros(n_trials, T_max);

for trial = 1:n_trials
    if mod(trial, 10) == 0
        fprintf('Trial %d / %d\n', trial, n_trials);
    end

    % generate sequence: pre N(0,I), post mixture
    pre  = randn(d, tau - 1);
    post = sample_post_fig2(d, T_max - (tau - 1));
    seq  = [pre, post];   % d x T_max

    traj_proposed(trial, :) = run_okcusum_traj(seq, X_blocks, N, w, sigma2, rho, T_max);
    traj_scanb(trial, :)    = run_scanb_traj(seq, ref_blocks_scanb, N, B_scanb, sigma, std_ZB, T_max);
end

% ── Compute mean & std ───────────────────────────────────────────────────
t_axis = 1:T_max;

mean_prop  = mean(traj_proposed, 1);
std_prop   = std(traj_proposed,  0, 1);

mean_scanb = mean(traj_scanb,    1);
std_scanb  = std(traj_scanb,     0, 1);

% ── Plot ─────────────────────────────────────────────────────────────────
figure('Position', [100 100 560 400]);
hold on;

% Light gray shading for pre-change region t < tau
pre_t = [0, tau, tau, 0];
pre_y = [0, 0, 1e6, 1e6];   % will be clipped by ylim
fill(pre_t, pre_y, [0.93 0.93 0.93], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

% Shaded std bands — Scan B (orange), Proposed (red)
fill([t_axis, fliplr(t_axis)], ...
     [mean_scanb + std_scanb, fliplr(max(mean_scanb - std_scanb, 0))], ...
     [1.0, 0.75, 0.4], 'EdgeColor', 'none', 'FaceAlpha', 0.5);

fill([t_axis, fliplr(t_axis)], ...
     [mean_prop + std_prop, fliplr(max(mean_prop - std_prop, 0))], ...
     [0.85, 0.35, 0.25], 'EdgeColor', 'none', 'FaceAlpha', 0.4);

% Mean lines
h_prop  = plot(t_axis, mean_prop,  '-', 'Color', [0.72, 0.07, 0.07], 'LineWidth', 2.0);
h_scanb = plot(t_axis, mean_scanb, '-', 'Color', [0.92, 0.58, 0.10], 'LineWidth', 2.0);

% Change-point dashed line
xline(tau, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2);

% Annotation
y_lim_top = max(max(mean_prop + std_prop), max(mean_scanb + std_scanb)) * 1.12;
text(tau - 2, y_lim_top * 0.28, sprintf('Change-point at t = %d', tau), ...
     'HorizontalAlignment', 'right', 'FontSize', 8.5, 'Color', [0.2 0.2 0.2]);

xlabel('Time t', 'FontSize', 11);
ylabel('Detection statistic', 'FontSize', 11);
legend([h_prop, h_scanb], {'Proposed', 'Scan B'}, 'Location', 'northwest', 'FontSize', 9);
xlim([0 T_max]);
ylim([0, y_lim_top]);
box on;

saveas(gcf, '/home/claude/fig2_trajectories.png');
fprintf('Saved fig2_trajectories.png\n');

% ═════════════════════════════════════════════════════════════════════════
%  LOCAL FUNCTIONS
% ═════════════════════════════════════════════════════════════════════════

function X = sample_post_fig2(d, n)
    % 0.3 * N(0, I)  +  0.7 * N(0, 4I)
    X = zeros(d, n);
    for i = 1:n
        if rand < 0.3
            X(:,i) = randn(d,1);
        else
            X(:,i) = 2 * randn(d,1);   % std=2 => variance=4
        end
    end
end

% ── Kernel helpers ────────────────────────────────────────────────────────

function K = kernel_matrix(A, B_mat, sigma2)
    K = exp(-pdist2(A', B_mat').^2 / (2*sigma2));
end

function val = k_func(x, y, sigma2)
    diff = x - y;
    val  = exp(-(diff'*diff) / (2*sigma2));
end

function val = h_func(x1, x2, y1, y2, sigma2)
    val = k_func(x1,x2,sigma2) + k_func(y1,y2,sigma2) ...
        - k_func(x1,y2,sigma2) - k_func(x2,y1,sigma2);
end

% ── Proposed: rho estimator ───────────────────────────────────────────────

function rho = estimate_rho(ref_data, N, sigma2, M)
    n_mc    = 500;
    h2_vals = zeros(n_mc, 1);
    for i = 1:n_mc
        idx = randi(M, 1, 4);
        hval = h_func(ref_data(:,idx(1)), ref_data(:,idx(2)), ...
                      ref_data(:,idx(3)), ref_data(:,idx(4)), sigma2);
        h2_vals(i) = hval^2;
    end
    E_h2 = mean(h2_vals);

    cov_vals = zeros(n_mc, 1);
    for i = 1:n_mc
        idx = randi(M, 1, 6);
        cov_vals(i) = h_func(ref_data(:,idx(1)), ref_data(:,idx(2)), ...
                              ref_data(:,idx(5)), ref_data(:,idx(6)), sigma2) * ...
                      h_func(ref_data(:,idx(3)), ref_data(:,idx(4)), ...
                              ref_data(:,idx(5)), ref_data(:,idx(6)), sigma2);
    end
    Cov_h = mean(cov_vals);

    inner = E_h2/N + (N-1)/N * Cov_h;
    rho   = 0.5 * inner^(-0.5);
end

% ── Proposed: trajectory — exact Algorithm 1, no early stopping ──────────
% The buffer is always size w. Valid statistics fire from t=w onward
% (first full window). With tau=51 and w=50, first stat is at t=50,
% then post-change data starts filling in from t=51 onwards.

function traj = run_okcusum_traj(seq, X_blocks, N, w, sigma2, rho, T_max)
    traj  = zeros(1, T_max);
    n_obs = size(seq, 2);
    d     = size(seq, 1);

    Y     = zeros(d, w);   % sliding window buffer

    % Precompute fixed G_XX for each reference block
    G_XX = cell(N, 1);
    for nn = 1:N
        G_XX{nn} = kernel_matrix(X_blocks{nn}, X_blocks{nn}, sigma2);
    end

    G_XY = cell(N, 1);
    for nn = 1:N
        G_XY{nn} = zeros(w, w);
    end
    G_YY = zeros(w, w);

    for t = 1:min(n_obs, T_max)
        Yt = seq(:, t);

        % Algorithm 1 lines 4-8: update window and Gram matrices
        Y(:, 1:w-1)   = Y(:, 2:w);
        Y(:, w)       = Yt;

        G_YY(1:w-1, 1:w-1) = G_YY(2:w, 2:w);
        new_col = kernel_matrix(Y, Yt, sigma2);
        G_YY(:, w) = new_col;
        G_YY(w, :) = new_col';

        for nn = 1:N
            G_XY{nn}(:, 1:w-1) = G_XY{nn}(:, 2:w);
            G_XY{nn}(:, w)     = kernel_matrix(X_blocks{nn}, Yt, sigma2);
        end

        if t < w
            continue;   % buffer not full yet
        end

        % Algorithm 1 lines 9: B=2 term
        z = 0;
        for nn = 1:N
            z = z + G_XX{nn}(w-1, w) + G_YY(w-1, w) ...
                  - G_XY{nn}(w-1, w) - G_XY{nn}(w, w-1);
        end
        %Zt = sqrt(2) * rho / N * z;
        Zt = 2 * rho / N * z;

        % Algorithm 1 lines 10-13: B=3..w terms (recursive z accumulation)
        for B = 3:w
            B_tilde = w - B + 1;
            for nn = 1:N
                for i = B_tilde+1 : w
                    z = z + G_XX{nn}(B_tilde, i) + G_YY(B_tilde, i) ...
                          - G_XY{nn}(B_tilde, i) - G_XY{nn}(i, B_tilde);
                end
            end
            %Zt = max(Zt, (2*rho) / (N * sqrt(B*(B-1))) * z);
            Zt = max(Zt, (2*sqrt(2)*rho) / (N * sqrt(B*(B-1))) * z);
        end

        traj(t) = max(Zt, 0);
    end
end

% ── Scan-B helpers ────────────────────────────────────────────────────────

function blocks = build_ref_blocks(ref_data, N, B, M)
    idx    = randperm(M, N*B);
    blocks = cell(N, 1);
    for n = 1:N
        cols      = idx((n-1)*B+1 : n*B);
        blocks{n} = ref_data(:, cols);
    end
end

function K = kernel_matrix_sigma(A, B_mat, sigma)
    K = exp(-pdist2(A', B_mat').^2 / (2*sigma^2));
end

function mmd_u = MMD_u(X, Y, sigma)
    K_XX = kernel_matrix_sigma(X, X, sigma);
    K_YY = kernel_matrix_sigma(Y, Y, sigma);
    K_XY = kernel_matrix_sigma(X, Y, sigma);
    B0   = size(X, 2);
    H    = K_XX + K_YY - K_XY - K_XY';
    H    = H - diag(diag(H));
    mmd_u = sum(H(:)) / (B0*(B0-1));
end

function D_hat = compute_D_hat(ref_blocks, Y, N, sigma)
    mmd_vals = zeros(N, 1);
    for n = 1:N
        mmd_vals(n) = MMD_u(ref_blocks{n}, Y, sigma);
    end
    D_hat = mean(mmd_vals);
end

function val = h_func_sigma(x1, x2, y1, y2, sigma)
    val = exp(-norm(x1-x2)^2/(2*sigma^2)) + exp(-norm(y1-y2)^2/(2*sigma^2)) ...
        - exp(-norm(x1-y2)^2/(2*sigma^2)) - exp(-norm(x2-y1)^2/(2*sigma^2));
end

function var_out = estimate_variance_scanb(ref_data, ref_blocks, N, B, sigma)
    n_mc    = 500;
    M_ref   = size(ref_data, 2);
    h2_vals = zeros(n_mc, 1);
    for i = 1:n_mc
        idx = randi(M_ref, 1, 4);
        hval = h_func_sigma(ref_data(:,idx(1)), ref_data(:,idx(2)), ...
                            ref_data(:,idx(3)), ref_data(:,idx(4)), sigma);
        h2_vals(i) = hval^2;
    end
    E_h2 = mean(h2_vals);

    cov_vals = zeros(n_mc, 1);
    for i = 1:n_mc
        idx = randi(M_ref, 1, 6);
        cov_vals(i) = h_func_sigma(ref_data(:,idx(1)), ref_data(:,idx(2)), ...
                                   ref_data(:,idx(5)), ref_data(:,idx(6)), sigma) * ...
                      h_func_sigma(ref_data(:,idx(3)), ref_data(:,idx(4)), ...
                                   ref_data(:,idx(5)), ref_data(:,idx(6)), sigma);
    end
    Cov_h = mean(cov_vals);   % E[h(x1,x2,y1,y2)*h(x3,x4,y1,y2)] — no subtraction

    C       = 2 / (B*(B-1));
    var_out = C * (E_h2/N + (N-1)/N * Cov_h);
end

% ── Scan-B: trajectory (no early stopping) ───────────────────────────────

function traj = run_scanb_traj(seq, ref_blocks, N, B, sigma, std_ZB, T_max)
    traj  = zeros(1, T_max);
    n_obs = size(seq, 2);

    for t = B:min(n_obs, T_max)
        Y     = seq(:, t-B+1:t);
        D_hat = compute_D_hat(ref_blocks, Y, N, sigma);
        traj(t) = D_hat / std_ZB;
    end
end