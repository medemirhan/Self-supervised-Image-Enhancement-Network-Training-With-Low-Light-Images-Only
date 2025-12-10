function falseColorImage = visualizeHsiFalseColor_indoor(HSI, waveStart_nm, waveEnd_nm, normalize, globalMax, globalMin)
    if nargin < 4
        normalize = 'none';
    end
    
    [~,~,bands] = size(HSI);
    wavelengths = linspace(waveStart_nm, waveEnd_nm, bands);
    
    % Find band indices closest to desired wavelengths
    [~, idxNIR] = min(abs(wavelengths - 800));
    [~, idxRed] = min(abs(wavelengths - 670));
    [~, idxGreen] = min(abs(wavelengths - 550));

    % Extract bands
    R = HSI(:, :, idxNIR);
    G = HSI(:, :, idxRed);
    B = HSI(:, :, idxGreen);
    
    maxx = max([max(R(:)), max(G(:)), max(B(:))]);
    
    if strcmp(normalize, 'zeroOne')
        % Normalize each band between 0 and 1
        R = mat2gray(R);
        G = mat2gray(G);
        B = mat2gray(B);
    elseif strcmp(normalize, 'divideMax')
        R = R/max(R(:));
        G = G/max(G(:));
        B = B/max(B(:));
    elseif strcmp(normalize, 'divideGlobalMax')
        R = R/globalMax;
        G = G/globalMax;
        B = B/globalMax;
    elseif strcmp(normalize, 'globalNorm')
        R = (R - globalMin) / (globalMax - globalMin);
        G = (G - globalMin) / (globalMax - globalMin);
        B = (B - globalMin) / (globalMax - globalMin);
    elseif strcmp(normalize, 'divideGMax')
        R = R/maxx;
        G = G/maxx;
        B = B/maxx;
    elseif strcmp(normalize, 'percClip')
        R = pctStretch(R,3,97,0);
        G = pctStretch(G,3,97,0);
        B = pctStretch(B,3,97,0);
        R = R/max(R(:));
        G = G/max(G(:));
        B = B/max(B(:));
    end

    % Combine into RGB image
    falseColorImage = cat(3, R, G, B);

    % Only plot if no output argument
    if nargout == 0
        figure;
        imshow(falseColorImage);
        title('False Color Composite (NIR-R-G) - ');
        clear falseColorImage; % Make sure no unwanted output happens
    end
end