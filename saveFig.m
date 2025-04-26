clear all
close all
clc

%% Settings
imname         = 'vegatables5';
imdir          = 'D:\results\comparison\low\';
algo           = 'low';
savepath       = 'C:\Users\medemirhan\Desktop\mdpi\figures\dataset';
%showBand       = 20;
waveStart_nm   = 453.8117;
waveEnd_nm     = 962.3318;
imname_postfix = '_falseColor';
key            = 'data';

%% Load and Prepare Image
impath = fullfile(imdir, strcat(imname, '.mat'));
dataStruct = load(impath);
data = dataStruct.(key);

% img  = data(:,:, showBand);
img = visualizeHsiFalseColor(data, waveStart_nm, waveEnd_nm);

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
