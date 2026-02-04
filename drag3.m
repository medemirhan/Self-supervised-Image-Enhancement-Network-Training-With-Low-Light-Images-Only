%% Multi-image HSI false-color + synchronized live spectra
clear all
close all
clc

mmname = '254';

%%
global whiteCal;
whiteCal = false;

%% ---- USER SETTINGS -------------------------------------------------------
% 1st image
IM(1).imdir  = 'D:\jyu\jyu_indoor\only_low_psnrs_64_aligned\high\train\';
IM(1).imname = mmname;
IM(1).key    = 'data';
IM(1).title  = 'gt';

% 2nd image
IM(2).imdir  = 'D:\jyu\jyu_indoor\only_low_psnrs_64_aligned\lowAligned\train\';
IM(2).imname = mmname;
IM(2).key    = 'data';
IM(2).title  = 'low';
% 
% % 3rd image
% IM(3).imdir  = 'D:\sslie\test_results_jyu_outdoor_64_registration_nonSaturated_splitted_v2_20250924_045002\';
% IM(3).imname = mmname;
% IM(3).key    = 'ref';
% IM(3).title  = 'enhanced';
% 
% % 3rd image
% IM(4).imdir  = 'D:\sslie\test_results_jyu_outdoor_64_registration_nonSaturated_splitted_v2_20250924_045002\artifacts\';
% IM(4).imname = mmname;
% IM(4).key    = 'ref';
% IM(4).title  = 'reflectance';

global whiteRegionTopXY;
global whitCalWinSize;
whitCalWinSize = [10,10];

if strcmp(mmname,'328')
    whiteRegionTopXY = [65,385];
elseif strcmp(mmname,'486')
    whiteRegionTopXY = [365,345];
elseif strcmp(mmname,'492')
    whiteRegionTopXY = [455,150];
end

% Wavelength range (must match the cubes)
% waveStart_nm = 397.32;
% waveEnd_nm   = 1003.58;
waveStart_nm = 414.63; % for 64-band
waveEnd_nm   = 985.05; % for 64-band

% Initial cursor (x,y) and local averaging window
x0           = 170; 
y0           = 117;
window_size  = 5;   % odd numbers are typical (3,5,7)

% for band index (1..C) on x-axis instead of wavelength, set false
use_wavelength_x = true;

%% ---- LOAD CUBES ----------------------------------------------------------
N = numel(IM);
datas   = cell(1,N);
imRGB   = cell(1,N);
sizesHW = zeros(N,2);
bandCounts = zeros(1,N);
HSI_cells = {};
for i = 1:N
    impath = fullfile(IM(i).imdir, [IM(i).imname '.mat']);
    S = load(impath);
    if ~isfield(S, IM(i).key)
        error('File %s does not contain variable "%s".', impath, IM(i).key);
    end
    datas{i} = S.(IM(i).key);
    
    [h,w,c] = size(datas{i});
    sizesHW(i,:) = [h,w];
    bandCounts(i) = c;
    
    if i == 4
        datas{i} = datas{i} * (4095 - 238) + 238;
    end
    
    %datas{i} = rot90(datas{i}, -1);
    
    HSI_cells{end+1} = datas{i};
end

visParams = computeFalseColorDisplayParams(HSI_cells, waveStart_nm, waveEnd_nm, 'pclip', 1, 99);
%visParams = computeFalseColorDisplayParams(HSI_cells, waveStart_nm, waveEnd_nm, 'fixedMax');

% Basic sanity checks
if any(bandCounts ~= bandCounts(1))
    error('All cubes must have the same number of bands. Got: %s', mat2str(bandCounts));
end
if any(sizesHW(:,1) ~= sizesHW(1,1)) || any(sizesHW(:,2) ~= sizesHW(1,2))
    warning('Cubes have different spatial sizes. This script assumes they are co-registered and same size.');
end

[h,w,C] = size(datas{1});
wavelengths = linspace(waveStart_nm, waveEnd_nm, C);

% Build false-color images (you can adapt your function if needed)
for i = 1:N
    % imRGB{i} = visualizeHsiFalseColor(datas{i}, waveStart_nm, waveEnd_nm, normalize);
    imRGB{i} = visualizeHsiFalseColor_SHARED(datas{i}, visParams, 0.85);
end

%% ---- IMAGE FIGURE WITH TILED LAYOUT -------------------------------------
% Choose a near-square tiling automatically
nCols = ceil(sqrt(N));
nRows = ceil(N / nCols);   % ensures nRows * nCols >= N

fig_img = figure('Name','HSI false-color (drag any cursor)','Color','w');
tiledlayout(fig_img, nRows, nCols, 'Padding','compact','TileSpacing','compact');

ax = gobjects(1,N);
pt = gobjects(1,N);
colors = lines(N);

for i = 1:N
    ax(i) = nexttile(i);
    imgShow = imRGB{i};

    imshow(imgShow, 'Parent', ax(i));
    title(ax(i), sprintf('%s', IM(i).title), 'Interpreter','none');
    hold(ax(i), 'on');

    % Clamp initial position into image bounds
    [x_init, y_init] = clampXY(x0, y0, w, h);

    % One point per image (kept in sync)
    pt(i) = drawpoint('Parent', ax(i), ...
        'Position', [x_init, y_init], ...
        'Color', colors(i,:), ...
        'Label', sprintf('(%d,%d)', x_init, y_init), ...
        'LabelVisible', 'on');
end

%% ---- SPECTRA FIGURE ------------------------------------------------------
fig_spec = figure('Name','Live spectra (same coord, all images)','Color','w');
ax_spec = axes('Parent', fig_spec); 
hold(ax_spec, 'on'); grid(ax_spec, 'on');

if use_wavelength_x
    xvals = wavelengths;
    xlabel(ax_spec, 'Wavelength (nm)');
else
    xvals = 1:C;
    xlabel(ax_spec, 'Band Index');
end
ylabel(ax_spec, 'Intensity');
%title(ax_spec, 'Drag any cursor; spectra update for ALL images');

% One line per image
ln = gobjects(1,N);
[y_init_spec, lbls] = computeAllSpectra(datas, x_init, y_init, window_size, IM);
for i = 1:N
    ln(i) = plot(ax_spec, xvals, y_init_spec(:,i), ...
        'LineWidth', 1.6, 'Color', colors(i,:), ...
        'DisplayName', lbls{i});
end
legend(ax_spec, 'Location','northeast');

%% ---- SHARED STATE & CALLBACKS -------------------------------------------
% We’ll keep minimal shared state in appdata on the image figure to avoid base evals.
state.datas        = datas;
state.IM           = IM;
state.window_size  = window_size;
state.colors       = colors;
state.ax_spec      = ax_spec;
state.lines        = ln;
state.points       = pt;
state.xvals        = xvals;
state.isUpdating   = false;  % to prevent recursive updates
setappdata(fig_img, 'state', state);

% Add listeners: dragging ANY point updates ALL points + ALL spectra
for i = 1:N
    addlistener(pt(i), 'MovingROI', @(src,evt) onMove(fig_img, src, evt, i, w, h));
    addlistener(pt(i), 'ROIMoved',  @(src,evt) onMove(fig_img, src, evt, i, w, h));
end

%% ====================== LOCAL FUNCTIONS ==================================

function [xC, yC] = clampXY(x, y, W, H)
    xC = max(1, min(W, round(x)));
    yC = max(1, min(H, round(y)));
end

function spec = getSpectrum(data, x, y, win)
    % Mean spectrum in a win x win patch around (x,y)
    [H, W, ~] = size(data);
    x = round(x); y = round(y);
    r = floor(win/2);
    xs = max(1, x - r);
    xe = min(W, x + r);
    ys = max(1, y - r);
    ye = min(H, y + r);
    patch = data(ys:ye, xs:xe, :);
    spec = squeeze(mean(mean(patch,1,'omitnan'),2,'omitnan'));  % Cx1
end

function [Y, labels] = computeAllSpectra(datas, x, y, win, IM)
    % Returns (C x N) matrix Y and labels for legend
    global whiteRegionTopXY;
    global whitCalWinSize;
    global whiteCal;
    Nloc = numel(datas);
    C = size(datas{1},3);
    Y = zeros(C, Nloc);
    labels = cell(1, Nloc);
    for k = 1:Nloc
        if whiteCal
            normData = normalizeByWhiteRegion(datas{k}, whiteRegionTopXY, whitCalWinSize);
            Y(:,k) = getSpectrum(normData, x, y, win);
        else
            Y(:,k) = getSpectrum(datas{k}, x, y, win);
        end
        labels{k} = sprintf('%s @ (%d,%d)', IM(k).title, x, y);
    end
end

function onMove(fig_img, src, evt, idxMoved, W, H)
    st = getappdata(fig_img, 'state');
    if st.isUpdating, return; end
    st.isUpdating = true;

    % The new position from the point being dragged
    pos = evt.CurrentPosition;
    [xC, yC] = clampXY(pos(1), pos(2), W, H);

    % Sync every point to the same (xC, yC) and update their labels
    for k = 1:numel(st.points)
        % Avoid unnecessary triggers: only set if changed
        pk = st.points(k).Position;
        if any(round(pk) ~= [xC, yC])
            st.points(k).Position = [xC, yC];
        end
        st.points(k).Label = sprintf('(%d,%d)', xC, yC);
    end

    % Recompute ALL spectra at this coord
    [Y, labels] = computeAllSpectra(st.datas, xC, yC, st.window_size, st.IM);

    % Update lines + legend labels
    for k = 1:numel(st.lines)
        set(st.lines(k), 'YData', Y(:,k), 'DisplayName', labels{k});
    end
    legend(st.ax_spec, 'Location','northeast'); drawnow;

    st.isUpdating = false;
    setappdata(fig_img, 'state', st);
end