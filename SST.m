%% Optimized SST Implementation
T = 800;
N = 5000;
Delta_1 = 0.5:0.5:5;
Delta = flip(1 ./Delta_1);
alpha = 0.01;
theta_0 = 0;
sigma = 1;
num_trials = 250;

ADD_sst = zeros(1, length(Delta));
falseAlarm = zeros(1, length(Delta));

% Pre-compute log terms to avoid repeated calculation
log_terms = zeros(1, N);
for t = 2:N
log_terms(t) = 2*log(2*(t-1)*sqrt(t+1)/alpha);
end

parfor k = 1:length(Delta)
fprintf('Processing delta index: %d/%d\n', k, length(Delta));
delta_val = Delta(k);
theta_1 = theta_0 + delta_val;


add_sum = 0;
detections = 0;

for trial = 1:num_trials
    % Generate data
    %X = [normrnd(theta_0, sigma, 1, T), normrnd(theta_1, sigma, 1, N-T)];

    %uniform
    a = theta_0 - 1;
    b = theta_0 + 1;
    X_pre = unifrnd(a, b, 1, T);
    a = theta_1 - 1;
    b = theta_1 + 1;
    X_post = unifrnd(a, b, 1, (N-T));

    X = [X_pre, X_post];  % Combine pre and post change data

    % Precompute cumulative sums for efficiency
    cumsum_X = [0, cumsum(X)];

    detected = false;
    detection_time = N;

    for t = 2:N
        % Vectorized mean calculations
        s_vals = 1:(t-1);
        means_1 = (cumsum_X(s_vals+1) - cumsum_X(1)) ./ s_vals;
        means_2 = (cumsum_X(t+1) - cumsum_X(s_vals+1)) ./ (t - s_vals);

        % Vectorized bound calculations
        b_joint = sigma * sqrt((1./s_vals + 1./(t-s_vals)) * (1 + 1/t) * log_terms(t));

        % Check detection condition
        if any(abs(means_1 - means_2) > b_joint)
            detected = true;
            detection_time = t;
            break;
        end
    end

    if detected && detection_time > T
        add_sum = add_sum + (detection_time - T);
        detections = detections + 1;
    elseif detected && detection_time <= T
        % This is a false alarm before the change point
    else
        % No detection occurred
        add_sum = add_sum + N;
    end
end

falseAlarm(k) = num_trials - detections;
if detections > 0
    ADD_sst(k) = add_sum / 250;
else
    ADD_sst(k) = inf;
end



end

%% Plot results
invDel = flip(1./Delta);
ADD_sst = flip(ADD_sst);
falseAlarm = flip(falseAlarm);

figure;
subplot(2,1,1);
plot(invDel, ADD_sst, 'r.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\Delta');
ylabel('Average Detection Delay');
title('ADD vs 1/\Delta for SST Detector');
grid on;

subplot(2,1,2);
plot(invDel, falseAlarm, 'b.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\Delta');
ylabel('Number of False Alarms');
title('False Alarms vs 1/\Delta for SST Detector');
grid on;