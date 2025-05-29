clear all
clc

%% Settings
imname         = '007_2_2021-01-20_024';
imdir          = 'D:\results\comparison\ours\';
algo           = 'ours';
savepath       = 'C:\Users\medemirhan\Desktop\jstsp\figures\results';
%showBand       = 20;
waveStart_nm   = 417.73399;
waveEnd_nm     = 976.35468;
imname_postfix = '_falseColor';
key            = 'ref';

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
