% Read the CSV (assuming no header)
data = readmatrix('d3h3.csv');  % Replace with your actual filename

% Extract time and Z-value
time = data(:, 1);
z = data(:, 2);

% Downsample factor
factor = 5;

% Downsample (keep every 5th sample)
time_20Hz = time(1:factor:end);
z_20Hz = z(1:factor:end);

% Combine into one matrix
z_20Hz=z_20Hz-mean(z_20Hz);
z_20Hz= z_20Hz/max(z_20Hz);
downsampled_data = [ z_20Hz];

% Write to a new CSV file
writematrix(downsampled_data, 'output_20Hz.csv');

fprintf('Downsampled Z-data saved to output_20Hz.csv\n');
