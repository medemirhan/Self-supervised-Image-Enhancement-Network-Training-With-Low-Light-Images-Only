function trueColorImage = visualizeHsiTrueColor(HSI, waveStart_nm, waveEnd_nm, imname, normalize)
    if nargin < 5
        normalize = false;
    end
    
    if nargin < 4
        imname = '';
    end
    
    [~,~,bands] = size(HSI);
    wavelengths = linspace(waveStart_nm, waveEnd_nm, bands);
    
    % Find nearest bands
    [~, idxR] = min(abs(wavelengths - 640));
    [~, idxG] = min(abs(wavelengths - 540));
    [~, idxB] = min(abs(wavelengths - 460));

    % Extract bands
    R = HSI(:, :, idxR);
    G = HSI(:, :, idxG);
    B = HSI(:, :, idxB);

    if normalize
        % Normalize between 0 and 1
        R = mat2gray(R);
        G = mat2gray(G);
        B = mat2gray(B);
    end

    % Stack into RGB image
    trueColorImage = cat(3, R, G, B);

    % Only plot if no output argument
    if nargout == 0
        figure;
        imshow(trueColorImage);
        title(['True Color Composite (R-G-B) - ', imname]);
        clear trueColorImage; % Make sure no unwanted output happens
    end
end


