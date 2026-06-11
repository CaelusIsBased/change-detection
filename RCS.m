% Optimized Repeated-FCS Detector Implementation
% ADD vs 1/DELTA using Monte Carlo

T = 800;
N = 5000;
Delta_1 = 0.5:0.5:5;
Delta = flip(1./Delta_1);
alpha = 0.01;
theta_0 = 0;
sigma = 1;
num_samples = 250;

ADD_FCS = zeros(1, length(Delta));
falseAlarm_FCS = zeros(1, length(Delta));

% Precompute constants
const_factor = 3.4 * sigma;
alpha_term = 0.72 * log(10.4/alpha);

parfor k = 1:length(Delta)
disp(['Processing delta index: ', num2str(k), ' of ', num2str(length(Delta))]);
del = Delta(k);


% Preallocate results for this delta
detection_times = zeros(1, num_samples);
detected_flags = false(1, num_samples);

theta_1 = theta_0 + del;

% Generate all samples upfront
%r1_all = normrnd(theta_0, sigma, num_samples, T);
%r2_all = normrnd(theta_1, sigma, num_samples, N-T);
%X_all = [r1_all, r2_all];  % [num_samples × N]

% Generate ALL samples at once (vectorized)
theta_1 = theta_0 + del;
a0 = theta_0 - 1;
b0 = theta_0 + 1;
a1 = theta_1 - 1;
b1 = theta_1 + 1;

% Generate all data upfront: [num_samples × N]
r1_all = unifrnd(a0, b0, num_samples, T);
r2_all = unifrnd(a1, b1, num_samples, N-T);
X_all = [r1_all, r2_all];  % [num_samples × N]

for s = 1:num_samples
    X = X_all(s, :);

    % Precompute cumulative sums for ALL possible subsequences
    % This is the KEY optimization: instead of recalculating means,
    % we use cumulative sums to get any range sum in O(1)
    cumsum_X = [0, cumsum(X)];  % Padded for easier indexing

    % Store current nested bounds for each CS
    % CS_lower(start_time) = current lower bound for CS starting at start_time
    % CS_upper(start_time) = current upper bound for CS starting at start_time
    CS_lower = -inf * ones(1, N);
    CS_upper = inf * ones(1, N);

    detected = false;
    detection_time = N;

    for n = 1:N
        % Update all active CSs (from start_time=1 to start_time=n)
        % Vectorize where possible

        start_times = 1:n;
        data_lengths = n - start_times + 1;

        % Vectorized mean calculation using cumulative sums
        % mean(X(start:n)) = (cumsum(n+1) - cumsum(start)) / length
        range_sums = cumsum_X(n+1) - cumsum_X(start_times);
        X_bar_vec = range_sums ./ data_lengths;

        % Vectorized confidence width calculation
        w_vec = const_factor * sqrt((log(log(2*sigma^2*data_lengths)) + alpha_term) ./ data_lengths);
        CS_l_vec = X_bar_vec - w_vec/2;
        CS_u_vec = X_bar_vec + w_vec/2;

        % Apply nesting property
        for start_time = start_times
            if n == start_time
                % First time for this CS
                CS_lower(start_time) = CS_l_vec(start_time);
                CS_upper(start_time) = CS_u_vec(start_time);
            else
                % Apply nesting: new interval must be subset of previous
                CS_lower(start_time) = max(CS_l_vec(start_time), CS_lower(start_time));
                CS_upper(start_time) = min(CS_u_vec(start_time), CS_upper(start_time));
            end
        end

        % Find intersection of all active CSs
        % This is vectorized: max/min over all active CSs
        overall_lower = max(CS_lower(1:n));
        overall_upper = min(CS_upper(1:n));

        % Check if intersection is empty
        if overall_lower > overall_upper
            detected = true;
            detection_time = n;
            break;
        end
    end

    % Record results
    detected_flags(s) = detected;
    if detected
        detection_times(s) = max(0, detection_time - T);
    else
        detection_times(s) = N;
    end
end

% Aggregate results
iter = sum(detected_flags);
falseAlarm_FCS(k) = num_samples - iter;

if iter > 0
    ADD_FCS(k) = sum(detection_times) / 250;
else
    ADD_FCS(k) = inf;
end



end

% Plot results
invDel = flip(1./Delta);
ADD_FCS = flip(ADD_FCS);
falseAlarm_FCS = flip(falseAlarm_FCS);

figure;
plot(invDel, ADD_FCS, 'b.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\delta');
ylabel('Average Detection Delay');
title('ADD vs 1/\delta for Repeated-FCS Detector (Optimized)');
grid on;

figure;
plot(invDel, falseAlarm_FCS, 'g.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\delta');
ylabel('Number of False Alarms');
title('False Alarms vs 1/\delta for Repeated-FCS Detector (Optimized)');
grid on;

% Display ARL bound check
expected_ARL = 1/alpha;
fprintf('Theoretical ARL lower bound: %.2f\n', expected_ARL);
fprintf('This should be around %.0f for alpha = %.3f\n', expected_ARL, alpha);