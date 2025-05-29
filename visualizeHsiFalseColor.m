function falseColorImage = visualizeHsiFalseColor(HSI, waveStart_nm, waveEnd_nm, imname, normalize)
    if nargin < 5
        normalize = false;
    end
    
    if nargin < 4
        imname = '';
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

    if normalize
        % Normalize each band between 0 and 1
        R = mat2gray(R);
        G = mat2gray(G);
        B = mat2gray(B);
%         R = R/max(R(:));
%         G = G/max(G(:));
%         B = B/max(B(:));
%        %--- percentile contrast stretch --------------------------------------
%        R = pctStretch(R,5,95,'true');
%        G = pctStretch(G,5,95,'true');
%        B = pctStretch(B,5,95,'true');
    end

    % Combine into RGB image
    falseColorImage = cat(3, R, G, B);

    % Only plot if no output argument
    if nargout == 0
        figure;
        imshow(falseColorImage);
        title(['False Color Composite (NIR-R-G) - ', imname]);
        clear falseColorImage; % Make sure no unwanted output happens
    end
end