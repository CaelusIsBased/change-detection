% Optimized BCS Detector Implementation
% ADD vs 1/DELTA BCS USING MONTE CARLO

T = 800;
N = 5000;
Delta_1 = 0.5:0.5:5;
Delta = flip(1 ./Delta_1);
alpha = 0.01;
theta_0 = 0;
sigma = 1;
num_samples = 250;

ADD_BC = zeros(1, length(Delta));
falseAlarm = zeros(1, length(Delta));

% Precompute constants
const_factor = 3.4 * sigma;
alpha_term = 0.72 * log(10.4/alpha);

parfor k = 1:length(Delta)
disp(['Processing delta index: ', num2str(k), ' of ', num2str(length(Delta))]);
del = Delta(k);


% Preallocate arrays for this delta value
detection_times = zeros(1, num_samples);
detected_flags = false(1, num_samples);

theta_1 = theta_0 + del;

% Generate all samples upfront
%r1_all = normrnd(theta_0, sigma, num_samples, T);
% = normrnd(theta_1, sigma, num_samples, N-T);

% Generate all data upfront: [num_samples × N]
a0 = theta_0 - 1;
b0 = theta_0 + 1;
a1 = theta_1 - 1;
b1 = theta_1 + 1;
r1_all = unifrnd(a0, b0, num_samples, T);
r2_all = unifrnd(a1, b1, num_samples, N-T);

X_all = [r1_all, r2_all];  % [num_samples × N]

% Process each sample
for s = 1:num_samples
    X = X_all(s, :);

    % Precompute cumulative sums and means for forward pass
    cumsum_X = cumsum(X);
    n_vec = (1:N)';

    % Forward CS bounds
    CS_l_nested = -inf;
    CS_u_nested = inf;

    detected = false;
    detection_time = N;

    for n = 2:N
        % ===== FORWARD CS (vectorized where possible) =====
        X_bar = cumsum_X(n) / n;
        wn = const_factor * sqrt((log(log(2*sigma^2*n)) + alpha_term) / n);
        CS_l = X_bar - wn/2;
        CS_u = X_bar + wn/2;

        % Apply nesting
        CS_l_nested = max(CS_l, CS_l_nested);
        CS_u_nested = min(CS_u, CS_u_nested);

        % ===== BACKWARD CS at time n (OPTIMIZED) =====
        % Key insight: We can compute cumulative sums of reversed sequence
        X_reverse = X(n:-1:1);
        cumsum_reverse = cumsum(X_reverse);

        % Vectorize the backward CS computation
        t_vec = (1:n)';
        X_bar_back_vec = cumsum_reverse' ./ t_vec;
        wt_back_vec = const_factor * sqrt((log(log(2*sigma^2*t_vec)) + alpha_term) ./ t_vec);

        bCS_l_vec = X_bar_back_vec - wt_back_vec/2;
        bCS_u_vec = X_bar_back_vec + wt_back_vec/2;

        % Apply nesting: take most restrictive bounds
        bCS_l_nested = max(bCS_l_vec);
        bCS_u_nested = min(bCS_u_vec);

        % ===== CHECK INTERSECTION =====
        if CS_u_nested < bCS_l_nested || bCS_u_nested < CS_l_nested
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
falseAlarm(k) = num_samples - iter;

if iter > 0
    ADD_BC(k) = sum(detection_times) / num_samples;
else
    ADD_BC(k) = inf;
end



end

% Plot results
invDel = flip(1./Delta);
ADD_BC = flip(ADD_BC);
falseAlarm = flip(falseAlarm);

figure;
loglog(invDel, ADD_BC, 'r.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\delta');
ylabel('Average Detection Delay');
title('ADD vs 1/\delta for BCS Detector (Optimized)');
grid on;

figure;
plot(invDel, falseAlarm, 'b.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\delta');
ylabel('Number of False Alarms');
title('False Alarms vs 1/\delta for BCS Detector (Optimized)');
grid on;