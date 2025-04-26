clear all
clc

%% Paths, keys, titles
dataToCompare = [
    "C:\Users\medemirhan\Desktop\n2n\PairLIE\data\label_ll\buildingblock.mat", ...
    "data", ...
    "original"; ...
    
    "C:\Users\medemirhan\Desktop\n2n\PairLIE\data\hsi_dataset\test\buildingblock.mat", ...
    "data", ...
    "low light"; ...
    
    "D:\sslie\test_results_ablation_cross_val_set1_refinedLoss_v4_20250404_012305\buildingblock.mat", ...
    "ref", ...
    "ours"; ...
    
    "C:\Users\medemirhan\Desktop\comparison\results\clahe\0.05\buildingblock.mat", ...
    "data", ...
    "clahe"; ...
    ];

%% Params
sampleBand = 20;
rows = [75, 180, 240, 300];
bands = [10, 15, 20, 25, 30, 35, 40, 45, 55];
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
    %imshow(curData(:,:,sampleBand));
    imshow(falseColorImg);
    if i==1
        hold on
        cols = [1, size(curData(:,:,sampleBand), 2)]; % Full width of the image
        for j=1:length(rows)
            line(cols, [rows(j), rows(j)], 'Color', 'r', 'LineWidth', 1.5);
            text(cols(1) + 5, rows(j), sprintf('Row %d', rows(j)), 'Color', 'y', 'FontSize', 10, 'FontWeight', 'bold');
        end
        hold off
    end
    title(curLabel);
end

%% Plot spectra
for i=1:length(rows)
    row = rows(i);
    figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
    for j=1:length(bands)
        lgnd = strings(numHsis,1);
        for k=1:numHsis
            curData = reshape(data(k,:,:,:), [h, w, c]);
            curData = curData(row, 1:end, bands(j));
            lgnd(k) = dataToCompare(k,3);
            subplot(3,3,j)
            plot(curData, 'linewidth', 1.1)
            hold on
        end
        hold off
        xlabel('x location');
        ylabel('value');
        title(['spectrum at row=' num2str(row) ' band=' num2str(bands(j))]);
        legend(lgnd);
    end
end

