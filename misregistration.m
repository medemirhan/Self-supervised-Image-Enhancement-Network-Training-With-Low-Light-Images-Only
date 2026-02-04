clear all
clc

check_misregistration_hsi( ...
  'D:\jyu\jyu_indoor\v2_64_aligned\low', ...
  'D:\jyu\jyu_indoor\v2_64_aligned\high', ...
  'D:\jyu\jyu_indoor\v2_64_aligned\misreg_report.csv', ...
  'D:\jyu\jyu_indoor\v2_64_aligned\previews', ...
  struct( ...
    'projMethod','median', ...   % 'band' | 'mean' | 'median' | 'pca'
    'bandForVis',30, ...        % used iff projMethod='band'
    'shiftThresh',1.0, ...
    'flowThresh',0.8, ...
    'saveAlignedDir','D:\jyu\jyu_indoor\v2_64_aligned\lowAligned' ... % '' to skip saving aligned cubes
  ));

function check_misregistration_hsi(lowDir, gtDir, outCsv, previewDir, opts)
%CHECK_MISREGISTRATION_HSI  Batch mis-registration checker for HSI .mat cubes.
% Each .mat contains variable 'data' of size [H W C].
%
% For each matching filename (case-insensitive) in lowDir and gtDir:
%   1) Load cubes (variable 'data').
%   2) Build 2D projections (lowProj, gtProj) from cubes for alignment:
%        opts.projMethod: 'band' (uses opts.bandForVis), 'mean', 'median', 'pca'
%   3) Estimate global translation (imregcorr) on projections.
%   4) Warp lowProj to GT; compute residual motion (optical flow or edge proxy).
%   5) Save metrics to CSV; optionally save preview images.
%   6) (Optional) Apply the same translation to EVERY BAND and save aligned low cube.
%
% Dependencies: Image Processing Toolbox; Computer Vision Toolbox (optional for optical flow).

% ----- options -----
if nargin < 5, opts = struct(); end
opts = defaults(opts, struct( ...
    'extensions', {'.mat'}, ...
    'projMethod', 'median', ...  % 'band' | 'mean' | 'median' | 'pca'
    'bandForVis', 1, ...
    'shiftThresh', 1.0, ...
    'flowThresh', 0.8, ...
    'resizeToMinSide', true, ...
    'maxPreviews', 30, ...
    'saveAlignedDir', '', ...    % '' to skip saving aligned cubes
    'verbose', true));

assert(isfolder(lowDir) && isfolder(gtDir), 'Input folders not found.');

filesLow = listFiles(lowDir, opts.extensions);
filesGT  = listFiles(gtDir,  opts.extensions);

% map by base filename (without extension), lowercase
mapLow = containers.Map('KeyType','char','ValueType','char');
for i=1:numel(filesLow)
    [~,b,~] = fileparts(filesLow{i});
    mapLow(lower(b)) = filesLow{i};
end

pairs = {};
for i=1:numel(filesGT)
    [~,b,~] = fileparts(filesGT{i});
    k = lower(b);
    if isKey(mapLow,k)
        pairs(end+1,:) = {mapLow(k), filesGT{i}}; %#ok<AGROW>
    end
end
if isempty(pairs)
    error('No matching .mat filenames between %s and %s', lowDir, gtDir);
end
if opts.verbose, fprintf('Found %d matching pairs.\n', size(pairs,1)); end

% results table
R = table('Size',[size(pairs,1), 9], ...
    'VariableTypes', {'string','string','double','double','double','double','double','logical','string'}, ...
    'VariableNames', {'file_low','file_gt','dx','dy','shift_px','mean_flow','p95_flow','flag_misaligned','projMethod'});

% ensure dirs
savePreviews = ~isempty(previewDir);
if savePreviews && ~isfolder(previewDir), mkdir(previewDir); end
saveAligned = ~isempty(opts.saveAlignedDir);
if saveAligned && ~isfolder(opts.saveAlignedDir), mkdir(opts.saveAlignedDir); end

for i=1:size(pairs,1)
    fLow = pairs{i,1}; fGT = pairs{i,2};
    try
        lowCube = loadHSICube(fLow);
        gtCube  = loadHSICube(fGT);

        % crop/resize to common size
        [h, w, ~] = sizeCommon(lowCube, gtCube);
        if opts.resizeToMinSide
            lowCube = imresize3nearest2D(lowCube, [h, w]);
            gtCube  = imresize3nearest2D(gtCube,  [h, w]);
        else
            lowCube = lowCube(1:h,1:w,:);
            gtCube  = gtCube(1:h,1:w,:);
        end

        % projection images for alignment
        lowProj = project2D(lowCube, opts);
        gtProj  = project2D(gtCube,  opts);

        % normalize to [0,1] to stabilize correlation
        lowProj = mat2gray(lowProj);
        gtProj  = mat2gray(gtProj);

        % global translation (illumination-robust)
        tform = imregcorr(lowProj, gtProj, 'translation'); % affine2d
        T  = tform.T; dx = T(3,1); dy = T(3,2);
        sp = hypot(dx, dy);

        % warp the *projection* for metric calc
        Rout  = imref2d(size(gtProj));
        lowW  = imwarp(lowProj, tform, 'OutputView', Rout, 'FillValues', median(lowProj(:)));

        % residual local motion (flow or edge proxy)
        [mFlow, p95Flow] = residualMotion(lowW, gtProj);

        flag = (sp > opts.shiftThresh) || (mFlow > opts.flowThresh);

        % preview
        if savePreviews && flag
            [~,base,~] = fileparts(fGT);
            outPNG = fullfile(previewDir, sprintf('%s_preview.png', base));
            savePreview(lowProj, gtProj, lowW, dx, dy, sp, mFlow, p95Flow, outPNG);
        end

        % optionally save aligned cube (apply SAME 2D shift to every band)
        if saveAligned
            alignedCube = imwarpCube2D(lowCube, tform, size(gtProj));
            [~,base,~] = fileparts(fLow);
            outMat = fullfile(opts.saveAlignedDir, [base '.mat']);
            data = single(alignedCube);
            save(outMat, 'data');
        end

        % record
        R{i,'file_low'} = string(relpath(fLow, lowDir));
        R{i,'file_gt'}  = string(relpath(fGT,  gtDir));
        R{i,'dx'}       = dx; R{i,'dy'} = dy; R{i,'shift_px'} = sp;
        R{i,'mean_flow'}= mFlow; R{i,'p95_flow'} = p95Flow;
        R{i,'flag_misaligned'} = flag;
        R{i,'projMethod'} = string(opts.projMethod);

        if opts.verbose && mod(i, max(1,round(size(pairs,1)/20)))==0
            fprintf('Processed %d/%d\n', i, size(pairs,1));
        end

    catch ME
        warning('Failed on pair %s | %s: %s', fLow, fGT, ME.message);
        R{i,'file_low'} = string(relpath(fLow, lowDir));
        R{i,'file_gt'}  = string(relpath(fGT,  gtDir));
        R{i,3:end-1} = {NaN, NaN, NaN, NaN, NaN, false};
        R{i,'projMethod'} = string(opts.projMethod);
    end
end

% sort & save
R = sortrows(R, {'flag_misaligned','p95_flow','shift_px'}, {'descend','descend','descend'});
writetable(R, outCsv);
if opts.verbose, fprintf('Saved report to %s\n', outCsv); end
if savePreviews, fprintf('Previews in %s (only flagged pairs).\n', previewDir); end
if saveAligned,  fprintf('Aligned low cubes in %s.\n', opts.saveAlignedDir); end

end % main

% ---------- helpers ----------

function cube = loadHSICube(f)
S = load(f, 'data');
if ~isfield(S, 'data'), error('File %s has no variable ''data''.', f); end
cube = S.data;
if ndims(cube) ~= 3, error('Variable ''data'' must be HxWxC.'); end
cube = im2double(cube);
end

function [h, w, c] = sizeCommon(A, B)
h = min(size(A,1), size(B,1));
w = min(size(A,2), size(B,2));
c = min(size(A,3), size(B,3)); %#ok<NASGU> % bands can differ; unused here
end

function out = imresize3nearest2D(cube, HW)
% Resize H and W for every band using nearest (avoids band mixing)
out = zeros([HW size(cube,3)], 'like', cube);
for k=1:size(cube,3)
    out(:,:,k) = imresize(cube(:,:,k), HW, 'nearest');
end
end

function img = project2D(cube, opts)
switch lower(opts.projMethod)
    case 'band'
        b = max(1, min(size(cube,3), round(opts.bandForVis)));
        img = cube(:,:,b);
    case 'mean'
        img = mean(cube, 3);
    case 'median'
        img = median(cube, 3);
    case 'pca'
        % PCA-1 projection
        [H,W,C] = size(cube);
        X = reshape(cube, [], C);
        X = X - mean(X,1);
        [U,~,~] = svd(X, 'econ');
        pc1 = U(:,1);
        img = reshape(pc1, H, W);
        img = rescale(img);
    otherwise
        error('Unknown projMethod: %s', opts.projMethod);
end
end

function [mMean, m95] = residualMotion(Aw, B)
% Try optical flow (Computer Vision Toolbox); else edge-distance proxy.
try
    A = imgaussfilt(Aw, 0.8);
    B = imgaussfilt(B,  0.8);
    of = opticalFlowLK('NoiseThreshold', 0.003);
    estimateFlow(of, A);
    f = estimateFlow(of, B);
    mag = hypot(f.Vx, f.Vy);
    mMean = mean(mag(:), 'omitnan');
    m95  = prctile(mag(:), 95);
catch
    [mMean, m95] = edgeMisalignProxy(Aw, B);
end
end

function [mMean, m95] = edgeMisalignProxy(I1w, I2)
E1 = edge(I1w, 'Canny'); E2 = edge(I2, 'Canny');
D1 = bwdist(E1); D2 = bwdist(E2);
d12 = D1(E2); d21 = D2(E1);
mMean = mean([d12(:); d21(:)], 'omitnan');
m95  = prctile([d12(:); d21(:)], 95);
end

function savePreview(Low, GT, LowW, dx, dy, sp, mFlow, p95Flow, outPNG)
E1  = edge(Low,  'Canny'); E1w = edge(LowW,'Canny'); E2 = edge(GT,'Canny'); %#ok<NASGU>
ov12  = overlayEdges(GT,  E1);
ov1w2 = overlayEdges(GT,  E1w);
diffImg = mat2gray(abs(LowW - GT));

h = figure('Visible','off','Color','w'); %#ok<NASGU>
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
nexttile; imshow(Low,[]);   title('Low proj');
nexttile; imshow(GT,[]);    title('GT proj');
nexttile; imshow(LowW,[]);  title(sprintf('Low warped (dx=%.2f, dy=%.2f)',dx,dy));
nexttile; imshow(ov12);     title('Edges: Low on GT');
nexttile; imshow(ov1w2);    title('Edges: Warped Low on GT');
nexttile; imshow(diffImg,[]); title(sprintf('Abs diff | shift=%.2f | meanFlow=%.2f | p95=%.2f', sp, mFlow, p95Flow));
exportgraphics(gcf, outPNG, 'Resolution', 150);
close(gcf);
end

function C = overlayEdges(I, E)
I = mat2gray(I);
C = repmat(I,1,1,3);
G = C(:,:,2);
G(E) = 1.0;
C(:,:,2) = G;
end

function rp = relpath(p, root)
p = string(p); root = string(root);
try rp = erase(p, string([char(root) filesep])); catch, rp = p; end
end

function L = listFiles(dirpath, exts)
if ischar(exts), exts = cellstr(exts); end
L = {};
for i=1:numel(exts)
    tmp = dir(fullfile(dirpath, ['**/*' exts{i}]));
    for j=1:numel(tmp)
        if ~tmp(j).isdir
            L{end+1} = fullfile(tmp(j).folder, tmp(j).name); %#ok<AGROW>
        end
    end
end
L = unique(L);
end

function S = defaults(S, D)
fn = fieldnames(D);
for i=1:numel(fn)
    k = fn{i};
    if ~isfield(S,k) || isempty(S.(k)), S.(k) = D.(k); end
end
end

function aligned = imwarpCube2D(cube, tform, targetHW)
% Apply a 2D affine2d tform to every band (same shift) and return HxWxC.
Rout = imref2d(targetHW);
aligned = zeros([targetHW size(cube,3)], 'like', cube);
fillVal = median(cube(:));
for k=1:size(cube,3)
    aligned(:,:,k) = imwarp(cube(:,:,k), tform, 'OutputView', Rout, 'FillValues', fillVal);
end
end
