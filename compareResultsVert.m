clear all
clc

%% Paths, keys, titles
dataToCompare = [
    "D:\results\comparison\normal\buildingblock.mat", ...
    "data", ...
    "Ground Truth"; ...

    "D:\results\comparison\low\buildingblock.mat", ...
    "data", ...
    "Low-light"; ...

    "D:\results\comparison\bm4d\buildingblock.mat", ...
    "data", ...
    "BM4D"; ...
    
    "D:\results\comparison\clahe\buildingblock.mat", ...
    "data", ...
    "CLAHE"; ...

%     "D:\results\comparison\deep_hs_prior\buildingblock.mat", ...
%     "pred", ...
%     "Deep HS Prior"; ...
    
    "D:\results\comparison\fast_hy_mix\buildingblock.mat", ...
    "data", ...
    "FastHyMix"; ...
    
    "D:\results\comparison\hcanet\buildingblock.mat", ...
    "data", ...
    "HCANet"; ...
    
    "D:\results\comparison\he\buildingblock.mat", ...
    "data", ...
    "HE"; ...
    
    "D:\results\comparison\lrtdtv\buildingblock.mat", ...
    "data", ...
    "LRTDTV"; ...
    
    "D:\results\comparison\mr\buildingblock.mat", ...
    "data", ...
    "MR"; ...
    
    "D:\results\comparison\msr\buildingblock.mat", ...
    "data", ...
    "MSR"; ...
    
    "D:\results\comparison\retinexnet\buildingblock.mat", ...
    "data", ...
    "RetinexNet"; ...
    
    "D:\results\comparison\ours\buildingblock.mat", ...
    "ref", ...
    "SS-HSLIE (Ours)"; ...    
    ];

%% Params
xLocs = [250, 110, 200, 250, 370, 100];
yLocs = [310, 190, 240, 170, 200, 65];
windowSize = 5; % must be odd number
sampleBand = 20;
waveStart_nm = 453.8117;
waveEnd_nm = 962.3318;

%% Load data
data = load(dataToCompare(1,1)).(dataToCompare(1,2));
[h,w,c] = size(data);
data = reshape(data, [1, h, w, c]);

numHsis = length(dataToCompare);
for i=2:numHsis
    curData = load(dataToCompare(i,1)).(dataToCompare(i,2));
    data = cat(1, data, reshape(curData, [1, h, w, c]));
end

%% Plot hsis
numCols = ceil(sqrt(numHsis));
numRows = ceil(numHsis / numCols);

figure;
for i=1:numHsis
    subplot(numRows, numCols, i);
    curData = reshape(data(i,:,:,:), [h, w, c]);
    curLabel = dataToCompare(i,3);
    falseColorImg = visualizeHsiFalseColor(curData, waveStart_nm, waveEnd_nm);
    % imshow(curData(:,:,sampleBand));
    imshow(falseColorImg);
    if i==1
        hold on
        plot(xLocs, yLocs, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
        for j = 1:length(xLocs)
            text(xLocs(j) + 5, yLocs(j) - 5, sprintf('[%d, %d]', xLocs(j), yLocs(j)), 'Color', 'y', 'FontSize', 10, 'FontWeight', 'bold');
        end
        hold off
    end
    title(curLabel);
end

%% Plot spectra
figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7]);
for i=1:length(xLocs)
    xLoc = xLocs(i);
    yLoc = yLocs(i);

    xStart = max(1, xLoc - floor(windowSize/2));
    xEnd = min(w, xStart + windowSize - 1);
    yStart = max(1, yLoc - floor(windowSize/2));
    yEnd = min(h, yStart + windowSize - 1);
    
    lgnd = strings(numHsis,1);
    subplot(2,3,i)
    for j=1:numHsis
        curData = reshape(data(j,:,:,:), [h, w, c]);
        lgnd(j) = dataToCompare(j,3);
        window = curData(yStart:yEnd, xStart:xEnd, :);
        spectrum = squeeze(sum(window, [1,2])) / (windowSize^2);
        plot(spectrum, 'linewidth', 1.1)
        hold on
    end
    hold off
    xlabel('band #');
    ylabel('value');
    title(['spectrum at [' num2str(xLoc) ',' num2str(yLoc) ']']);
    legend(lgnd);
end



