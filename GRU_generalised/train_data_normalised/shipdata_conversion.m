% File: process_csv.m

% Read the full-length CSV file
inputFile = 'D1H3_normalised.csv';        % <-- Change this to your actual input file name
outputFile = 'Heave_ship.csv';      % <-- Output CSV file

inputData = readmatrix(inputFile);  % Assumes numeric data

% Create an output matrix with 4 columns
nRows = length(inputData);
outputData = zeros(nRows, 4);       % Initialize all zeros

% Assign the input data to the 3rd column
outputData(:, 3) = inputData;

% Write the output matrix to a new CSV
writematrix(outputData, outputFile);
