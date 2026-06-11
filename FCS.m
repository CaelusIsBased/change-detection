% Optimized BCS Detector Implementation
% ADD vs 1/DELTA BCS USING MONTE CARLO

T = 800;
N = 5000;
Delta_1 = 0.5:0.5:5;
Delta = flip(1 ./Delta_1);
alpha = 0.01;
theta_0 = 0;
sigma = 1;
ADD_FC = zeros(1, length(Delta));
falseAlarm = zeros(1, length(Delta));

for k = 1:length(Delta)
disp(['Processing delta index: ', num2str(k)]);
del = Delta(k);
add_fc = 0;
iter = 0;


for s = 1:250
    theta_1 = theta_0 + del;

    % Generate data
    %r1 = normrnd(theta_0, sigma, [1, T]);
    %r2 = normrnd(theta_1, sigma, [1, (N-T)]);

    %uniform
    a = theta_0 - 1;
    b = theta_0 + 1;
    r1 = unifrnd(a, b, 1, T);
    a = theta_1 - 1;
    b = theta_1 + 1;
    r2 = unifrnd(a, b, 1, (N-T));

    X = cat(2, r1, r2);

    % Initialize Forward CS bounds
    CS_l_prev = -inf;
    CS_u_prev = inf;

    detected = false;
    detection_time = N;

    for n = 1:N
        % ===== FORWARD CS =====
        X_bar = mean(X(1:n));
        wn = 3.4*sigma*sqrt((log(log(2*sigma^2*n)) + 0.72*log(10.4/alpha))/n);
        CS_l = X_bar - wn/2;
        CS_u = X_bar + wn/2;

        % Apply nesting for forward CS
        CS_l_current = max(CS_l, CS_l_prev);
        CS_u_current = min(CS_u, CS_u_prev);

        % ===== CHECK INTERSECTION =====

        if CS_l_current > CS_u_current
            detected = true;
            detection_time = n;
            break;
        end

        % Update for next iteration
        CS_l_prev = CS_l_current;
        CS_u_prev = CS_u_current;
    end

    % Record results
    if detected
        add_fc = add_fc + max(0, detection_time - T);
        iter = iter + 1;

    else
        add_fc = add_fc + N;
        %iter = iter + 1;
    end
end

falseAlarm(k) = 250 - iter;
if iter > 0
    ADD_FC(k) = add_fc / 250;
else
    ADD_FC(k) = inf; % No detections occurred
end



end

% Plot results
invDel = flip(1./Delta);
ADD_FC = flip(ADD_FC);
falseAlarm = flip(falseAlarm);

figure;
plot(invDel, ADD_FC, 'r.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\delta');
ylabel('Average Detection Delay');
title('ADD vs 1/\delta for FCS Detector (Optimized)');
grid on;

figure;
plot(invDel, falseAlarm, 'b.-', 'LineWidth', 2, 'MarkerSize', 10);
xlabel('1/\delta');
ylabel('Number of False Alarms');
title('False Alarms vs 1/\delta for BCS Detector (Optimized)');
grid on;