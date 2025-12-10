clear all
close all
clc

%% Settings
imname         = 'buildingblock';
imdir          = 'D:\results\comparison\enlightengan\indoor';
algo           = 'enlightengan';
savepath       = 'C:\Users\medemirhan\Desktop\jstsp_versions\revision_2\figures\results';
waveStart_nm   = 453.8117;
waveEnd_nm     = 962.3318;
% waveStart_nm   = 414.63; % for jyu
% waveEnd_nm     = 985.05; % for jyu
imname_postfix = '_falseColor';
key            = 'data';

%% Load and Prepare Image
impath = fullfile(imdir, strcat(imname, '.mat'));
dataStruct = load(impath);
data = dataStruct.(key);

img = visualizeHsiFalseColor_indoor(data, waveStart_nm, waveEnd_nm);
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
