clear all
close all
clc

env = 'outdoor';

if strcmp(env, 'indoor')
    imname         = 'vegatables5';
    imdir          = 'D:\results\comparison\normal\indoor\';
    waveStart_nm   = 453.81;
    waveEnd_nm     = 962.33;
    x_locs = [100, 110, 200, 370];
    y_locs = [65, 190, 240, 200];
    
elseif strcmp(env, 'outdoor')
    imname         = '007_2_2021-01-20_024';
    imdir          = 'D:\results\comparison\normal\outdoor\';
    waveStart_nm   = 407.73399;
    waveEnd_nm     = 976.35468;
    x_locs = [75, 160, 250, 425, 287, 345];
    y_locs = [175, 340, 325, 420, 357, 357];
    
else 
    error('undefined env');
end

key = 'data';
window_size = 5;

% Load HSI data
impath = fullfile(imdir, strcat(imname, '.mat'));
dataStruct = load(impath);
data = dataStruct.(key);
[h, w, c] = size(data);

% Compute wavelength array
wavelengths = linspace(waveStart_nm, waveEnd_nm, c);

% Show false-color image
img = visualizeHsiFalseColor(data, waveStart_nm, waveEnd_nm);

fig_img = figure();
imshow(img);
ax_img = gca; 
hold(ax_img, 'on');

title(ax_img, 'Drag points to update spectra');


% Create spectra figure
fig_spectra = figure();
spectraAxes = axes(fig_spectra);
hold(spectraAxes, 'on');
xlabel(spectraAxes, 'Band Index');
ylabel(spectraAxes, 'Intensity');
grid(spectraAxes, 'on');
title(spectraAxes, 'Live Spectra');


% Set up visuals
numPoints = length(x_locs);
colors     = lines(numPoints);
letters    = 'a':'z';

% Preallocate handles
pointHandles = gobjects(numPoints, 1);
plotHandles  = gobjects(numPoints, 1);

% Initial legend labels
legend_labels = strings(1, numPoints);

% Draw initial points & spectra
for idx = 1:numPoints
    x = x_locs(idx);
    y = y_locs(idx);

    % Create draggable point, explicitly on ax_img
    pointHandles(idx) = drawpoint('Parent', ax_img, ...
                                  'Position', [x, y], ...
                                  'Color', colors(idx, :), ...
                                  'Label', letters(idx), ...
                                  'LabelVisible', 'on');

    % Compute spectrum at (x,y)
    spectrum = getSpectrum(data, x, y, window_size);

    % Plot spectrum in the separate figure
    figure(fig_spectra);
    plotHandles(idx) = plot(wavelengths, spectrum, ...
                            'Color', colors(idx, :), ...
                            'LineWidth', 1.5, ...
                            'DisplayName', sprintf('%s: (%d, %d)', letters(idx), x, y));
    
    legend_labels(idx) = sprintf('%s: (%d, %d)', letters(idx), x, y);

    % Add listener so that dragging updates its spectrum
    addlistener(pointHandles(idx), 'MovingROI', ...
        @(src, evt) updateSpectrum(src, evt, idx));
end

% Finally create legend once
legend(spectraAxes, legend_labels);


% === Helper: extract the mean spectrum around (x,y) ===
function spectrum = getSpectrum(data, x, y, win)
    [h, w, ~] = size(data);
    x = round(x); 
    y = round(y);

    x_start = max(1, x - floor(win/2));
    x_end   = min(w, x_start + win - 1);
    y_start = max(1, y - floor(win/2));
    y_end   = min(h, y_start + win - 1);

    patch = data(y_start:y_end, x_start:x_end, :);
    spectrum = squeeze(mean(mean(patch, 1), 2));  % (c × 1)
end


% === Callback: called while a point is being dragged ===
function updateSpectrum(src, evt, idx)
    pos = round(evt.CurrentPosition);
    x = pos(1);
    y = pos(2);

    % Retrieve shared variables from base workspace
    data        = evalin('base', 'data');
    window_size = evalin('base', 'window_size');
    wavelengths = evalin('base', 'wavelengths');
    plotHandles = evalin('base', 'plotHandles');

    % Compute new spectrum
    spectrum = getSpectrum(data, x, y, window_size);

    % Update that line’s YData
    set(plotHandles(idx), 'YData', spectrum);

    % Also update its legend entry
    letters   = evalin('base', 'letters');
    newLabel  = sprintf('%s: (%d, %d)', letters(idx), x, y);
    set(plotHandles(idx), 'DisplayName', newLabel);

    % Refresh the legend
    figure(evalin('base', 'fig_spectra'));
    legend(evalin('base', 'spectraAxes'));
end
