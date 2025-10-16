%PROCESS_DATA_WITH_MEKF_MODEL Processes 1D data using a Kalman filter
%   with a constant acceleration model, analogous to the C++ MEKF's
%   translational dynamics.
%
%   Args:
%       inDir (string): Directory containing input CSV files.
%       outDir (string): Directory to save output CSV files with velocity.
%       plotDir (string): Directory to save comparison plots.

% --- Setup ---
inDir = "train_data_normalised"; 
outDir = "train_data_mekf_vel";
plotDir = "mekf_comparison_plots";

if ~exist(outDir, 'dir'), mkdir(outDir); end
if ~exist(plotDir, 'dir'), mkdir(plotDir); end

% --- System & Filter Parameters ---
FS = 20;        % Sampling frequency in Hz
DT = 1/FS;      % Time step in seconds

% State Vector: x = [position; velocity; acceleration]
% This matches the translational part of the C++ MEKF model.
% x_k = F * x_{k-1}
F = [1, DT, 0.5*DT^2;
     0,  1,       DT;
     0,  0,        1];

% Measurement Model: We only measure position.
% z_k = H * x_k
H = [1, 0, 0];

% --- Tuning Parameters (Crucial for good performance) ---
% These values are analogous to Q and R in the C++ code.

% Process Noise Covariance (Q): Uncertainty in our model.
% We assume acceleration is not perfectly constant and can change suddenly (jerk).
sigma_j = 0.5; % Standard deviation of jerk (tune this value)
G = [DT^3/6; DT^2/2; DT]; % Noise gain matrix
Q = G * G' * sigma_j^2;

% Measurement Noise Covariance (R): How much we trust the measurement.
sigma_z = 0.02; % Standard deviation of measurement noise (in normalized units)
R = sigma_z^2;

% --- Main Processing Loop ---
files = dir(fullfile(inDir, "D1H*_normalised.csv"));
fprintf('Found %d files to process...\n', numel(files));

for k = 1:numel(files)
    % --- Load Data ---
    inPath = fullfile(files(k).folder, files(k).name);
    data = readmatrix(inPath);
    z_meas = data(:,1); % The measured Z data

    % --- Kalman Filter Initialization ---
    x = [z_meas(1); 0; 0];      % Initial state [z_0, vz_0, az_0]
    P = diag([sigma_z^2, 1, 1]); % Initial covariance, small pos uncertainty, high vel/accel uncertainty

    % Preallocate arrays for results
    N = numel(z_meas);
    z_hat   = zeros(N, 1);
    vz_hat  = zeros(N, 1);
    az_hat  = zeros(N, 1);

    % --- Run the Filter ---
    for i = 1:N
        % 1. Predict
        x_pred = F * x;
        P_pred = F * P * F' + Q;

        % 2. Update
        y = z_meas(i) - H * x_pred;     % Innovation (measurement residual)
        S = H * P_pred * H' + R;        % Innovation covariance
        K = P_pred * H' / S;            % Kalman Gain
        x = x_pred + K * y;             % Update state estimate
        P = (eye(3) - K * H) * P_pred;  % Update covariance

        % Store results
        z_hat(i)  = x(1);
        vz_hat(i) = x(2);
        az_hat(i) = x(3);
    end

    % --- Calculate Velocity via Differentiation for Comparison ---
    % Use gradient on the SMOOTHED position data for a fairer comparison
    vz_diff = gradient(z_hat, DT);

    % --- Save Results to CSV (no headers, only z & v_hat) ---
    data = [z_meas(:), vz_hat(:)];             % two columns: z, v_hat
    [~, baseName, ~] = fileparts(files(k).name);
    outPath = fullfile(outDir, [baseName, '_with_mekf_vel.csv']);
    writematrix(data, outPath);                 % no variable names written
    fprintf('✓ Wrote %s\n', outPath);
    % 
    % % --- Plotting ---
    % fig = figure('Visible', 'off');
    % length=1000;
    % time = (0:length-1) * DT;
    % figure;
    % % Plot position on left y-axis
    % yyaxis left
    % plot(time, z_meas(1:length), '.', 'Color', [0.7 0.7 0.7], 'DisplayName', 'Raw Z Data');
    % hold on;
    % plot(time, z_hat(1:length), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Filtered Z (KF)');
    % ylabel('Position (normalized units)');
    % ylim('padded');
    % 
    % % Plot velocities on right y-axis
    % yyaxis right
    % plot(time, vz_hat(1:length), 'r-', 'LineWidth', 2, 'DisplayName', 'Velocity (KF)');
    % hold on;
    % plot(time, vz_diff(1:length), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Velocity (Gradient)');
    % ylabel('Velocity (norm. units / s)');
    % ylim('padded');
    % 
    % % Finalize plot
    % grid on;
    % title(sprintf('Kalman Filter vs. Differentiation for %s', baseName), 'Interpreter', 'none');
    % xlabel('Time (s)');
    % legend('show', 'Location', 'best');
    % hold off;
    % 
    % % Save the plot
    % plotPath = fullfile(plotDir, [baseName, '_plot.png']);
    % saveas(fig, plotPath);
    % fprintf('✓ Saved plot %s\n', plotPath);
    % % close(fig);
end

fprintf('\nProcessing complete.\n');
