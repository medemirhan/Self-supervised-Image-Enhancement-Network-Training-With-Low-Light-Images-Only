clear all
close all
clc

%% Settings
imname         = 'buildingblock';
imdir          = 'D:\results\comparison\exposure_diffusion\indoor';
algo           = 'exposure_diffusion';
savepath       = 'C:\Users\medemirhan\Desktop\tez\latex\figures\results';
% waveStart_nm   = 407.73399;
% waveEnd_nm     = 976.35468;
waveStart_nm   = 397.32;
waveEnd_nm     = 1003.58;
imname_postfix = '_falseColor';
key            = 'data';

%% Load and Prepare Image
impath = fullfile(imdir, strcat(imname, '.mat'));
dataStruct = load(impath);
data = dataStruct.(key);

img = visualizeHsiFalseColor(data, waveStart_nm, waveEnd_nm);
% figure, imshow(img)

% img = visualizeHsiFalseColor(data, waveStart_nm, waveEnd_nm, 'divideMax');
% figure, imshow(img)

% img = visualizeHsiFalseColor(data, waveStart_nm, waveEnd_nm, 1157, 238, 'globalNorm');

% img = visualizeHsiFalseColor(data, waveStart_nm, waveEnd_nm, 'divideGlobalMax', 1157);

% title(algo)

% Get image dimensions
% [h, w] = size(img);
[h, w, ~] = size(img);

%% Create Figure with Exact Image Dimensions
% Using 'pixels' for the axes ensures that the axes exactly match the image's pixel dimensions.
fig = figure('Units', 'pixels', 'Position', [100 100 w h], 'Color', 'white');
ax  = axes('Units', 'pixels', 'Position', [0 0 w h]); % Set position in pixels

imshow(img, 'Border', 'tight', 'InitialMagnification', 100);
axis off;
drawnow;  % Ensure the figure is rendered correctly before export

%% Export the Image without Cropping
% Use fullfile for robust path construction.
exportgraphics(ax, fullfile(savepath, strcat(imname, '_', algo,...
    imname_postfix, '.eps')), 'ContentType', 'image', 'Resolution', 96);
