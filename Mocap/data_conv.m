% data = readtable("D1H1.csv",VariableNamingRule="preserve");
% heave= data.platform_heave;
% a=mean(heave)
% Loop over files D1H1.csv to D1H5.csv

for i = 1:5
    % Build filename
    filename = sprintf('D1H%d.csv', i);
    
    % Load CSV as table, preserve column names exactly
    data = readtable(filename, VariableNamingRule="preserve");
    
    % Extract 'platform_heave' column
    heave = data.platform_heave;
    
    % Normalise to [-1, 1]
    heave_min = min(heave);
    heave_max = max(heave);
    heave_normalised = 2 * (heave - heave_min) / (heave_max - heave_min) - 1;
    
    % Convert to table with clear name
    normalised_table = table(heave_normalised, ...
        'VariableNames', {'platform_heave_normalised'});
    
    % Save just the normalised column
    out_filename = sprintf('D1H%d_normalised.csv', i);
    writetable(normalised_table, out_filename);
    
    % Status
    fprintf('Saved %s\n', out_filename);
end
