% BRAVO-CS Detector Implementation
% Based on "Quickest Mean-Change Detection via Confidence Sequence and Backward Sample Averages"

%% Parameters
T = 800;                    % Change point
N = 2000;                   % Total observations
invDel = 0.5:0.5:5;      % 1/Delta values
Delta = 1 ./ invDel;     % Delta values
alpha = 0.01;               % Significance level
theta_0 = 0;                % Pre-change mean
sigma = 1;                  % Standard deviation
num_trials = 250;           % Monte Carlo trials

% Initialize results
ADD_BR = zeros(1, length(Delta));
detection_count = zeros(1, length(Delta));

%% Main simulation loop
for k = 1:length(Delta)
del = Delta(k);
theta_1 = theta_0 + del;


total_add = 0;
valid_detections = 0;

for trial = 1:num_trials
    % Generate data
    %X_pre = normrnd(theta_0, sigma, [1, T]);
    %X_post = normrnd(theta_1, sigma, [1, N-T]);

    %uniform
    a = theta_0 - 1;
    b = theta_0 + 1;
    X_pre = unifrnd(a, b, 1, T);
    a = theta_1 - 1;
    b = theta_1 + 1;
    X_post = unifrnd(a, b, 1, (N-T));

    % Combine pre-change and post-change data
    X = [X_pre, X_post];

    % Run BRAVO-CS Detector
    tau_bravo = run_bravo_cs(X, alpha, sigma, N);

    if tau_bravo <= N
        detection_delay = max(0, tau_bravo - T);
        total_add = total_add + detection_delay;
        valid_detections = valid_detections + 1;
    end
end

% Calculate average detection delay
if valid_detections > 0
    ADD_BR(k) = total_add / valid_detections;
else
    ADD_BR(k) = NaN;
end

detection_count(k) = valid_detections;

fprintf('Delta = %.2f (1/Delta = %.1f): Detections = %d/%d, ADD = %.2f\\n', ...
    del, invDel(k), valid_detections, num_trials, ADD_BR(k));



end

%% Plot results
figure('Position', [100, 100, 800, 600]);
plot(invDel, ADD_BR, 'r.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\Delta', 'FontSize', 14);
ylabel('Average Detection Delay', 'FontSize', 14);
title('BRAVO-CS: ADD vs 1/\Delta under Gaussian Distributions', 'FontSize', 16);
grid on;
set(gca, 'FontSize', 12);

% Plot detection rates
figure('Position', [200, 200, 800, 600]);
plot(invDel, detection_count/num_trials * 100, 'b.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\Delta', 'FontSize', 14);
ylabel('Detection Rate (%)', 'FontSize', 14);
title('BRAVO-CS: Detection Rate vs 1/\Delta', 'FontSize', 16);
grid on;
set(gca, 'FontSize', 12);
ylim([0, 100]);

%% BRAVO-CS Detector Implementation
function tau = run_bravo_cs(X, alpha, sigma, N)
tau = N + 1; % Default: no detection within observation window


% Initialize variables for nested CS
CS_lower_prev = -Inf;
CS_upper_prev = Inf;

for n = 1:N
    % Compute confidence sequence width according to Example 1 in paper
    if n == 1
        % Handle the case when log(log(2*sigma^2*n)) might be undefined
        wn = 3.4 * sigma * sqrt(0.72 * log(10.4/alpha));
    else
        log_term = log(log(2*sigma^2*n));
        if log_term <= 0
            log_term = 0.01; % Small positive value to avoid issues
        end
        wn = 3.4 * sigma * sqrt((log_term + 0.72*log(10.4/alpha))/n);
    end

    % Compute sample mean (center of forward CS)
    X_bar = mean(X(1:n));

    % Compute current CS bounds
    CS_lower_curr = X_bar - wn/2;
    CS_upper_curr = X_bar + wn/2;

    % Apply nesting property: intersection with previous CS
    CS_lower = max(CS_lower_curr, CS_lower_prev);
    CS_upper = min(CS_upper_curr, CS_upper_prev);

    % Update for next iteration
    CS_lower_prev = CS_lower;
    CS_upper_prev = CS_upper;

    % Compute backward sample average (BSA)
    if n == 1
        X_hat = X(1);
    else
        % BSA is average of second half of observations
        start_idx = floor(n/2) + 1;
        X_hat = mean(X(start_idx:n));
    end

    % Check if BSA escapes the nested CS
    if X_hat < CS_lower || X_hat > CS_upper
        tau = n;
        break;
    end

    % Safety check: if CS becomes empty (shouldn't happen with proper implementation)
    if CS_lower > CS_upper
        % This indicates numerical issues - use current bounds instead
        CS_lower = CS_lower_curr;
        CS_upper = CS_upper_curr;
        CS_lower_prev = CS_lower;
        CS_upper_prev = CS_upper;
    end
end



end

%% Visualization function for single run (optional)
function visualize_bravo_cs(X, alpha, sigma, T)
N = length(X);


% Arrays to store values for plotting
X_hat_vec = zeros(1, N);
CS_lower_vec = zeros(1, N);
CS_upper_vec = zeros(1, N);
X_bar_vec = zeros(1, N);

% Initialize nested CS bounds
CS_lower_prev = -Inf;
CS_upper_prev = Inf;
detection_point = N + 1;

for n = 1:N
    % Compute CS width
    if n == 1
        wn = 3.4 * sigma * sqrt(0.72 * log(10.4/alpha));
    else
        log_term = max(log(log(2*sigma^2*n)), 0.01);
        wn = 3.4 * sigma * sqrt((log_term + 0.72*log(10.4/alpha))/n);
    end

    % Sample mean
    X_bar = mean(X(1:n));
    X_bar_vec(n) = X_bar;

    % Current CS bounds
    CS_lower_curr = X_bar - wn/2;
    CS_upper_curr = X_bar + wn/2;

    % Nested CS bounds
    CS_lower = max(CS_lower_curr, CS_lower_prev);
    CS_upper = min(CS_upper_curr, CS_upper_prev);

    CS_lower_vec(n) = CS_lower;
    CS_upper_vec(n) = CS_upper;

    % Update for next iteration
    CS_lower_prev = CS_lower;
    CS_upper_prev = CS_upper;

    % BSA
    if n == 1
        X_hat = X(1);
    else
        start_idx = floor(n/2) + 1;
        X_hat = mean(X(start_idx:n));
    end
    X_hat_vec(n) = X_hat;

    % Check detection
    if (X_hat < CS_lower || X_hat > CS_upper) && detection_point > N
        detection_point = n;

    else

        detection_point = N;
    end
end

% Plot
figure('Position', [300, 300, 1000, 600]);
hold on;

% Plot raw data (thinned for clarity)
thin_factor = max(1, floor(N/200));
plot(1:thin_factor:N, X(1:thin_factor:N), 'k.', 'MarkerSize', 4, 'DisplayName', 'Observations');

% Plot means and CS
plot(1:N, X_bar_vec, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Sample Mean');
plot(1:N, X_hat_vec, 'r-', 'LineWidth', 2, 'DisplayName', 'Backward Sample Average');
plot(1:N, CS_lower_vec, 'g--', 'LineWidth', 1.5, 'DisplayName', 'CS Lower Bound');
plot(1:N, CS_upper_vec, 'g--', 'LineWidth', 1.5, 'DisplayName', 'CS Upper Bound');

% Mark change point and detection
plot([T, T], ylim, 'm-', 'LineWidth', 2, 'DisplayName', 'True Change Point');
if detection_point <= N
    plot([detection_point, detection_point], ylim, 'r-', 'LineWidth', 2, 'DisplayName', 'Detection Point');
end

xlabel('Time Step', 'FontSize', 14);
ylabel('Value', 'FontSize', 14);
title(sprintf('BRAVO-CS Detection Example (T=%d, Detection=%d)', T, detection_point), 'FontSize', 16);
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 12);



end

%% Example usage of visualization (uncomment to use)
% Generate example data for visualization
% X_example = [normrnd(0, 1, [1, T]), normrnd(0.7, 1, [1, N-T])];
% visualize_bravo_cs(X_example, alpha, sigma, T);