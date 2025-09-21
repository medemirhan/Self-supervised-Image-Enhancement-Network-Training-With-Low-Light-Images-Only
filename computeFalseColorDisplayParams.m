function params = computeFalseColorDisplayParams(HSI_cells, waveStart_nm, waveEnd_nm, method, pLow, pHigh)
% HSI_cells: cell array of HSI cubes {HSI1, HSI2, ...}
% method: 'fixedMax' or 'pclip'
% pLow/pHigh: percentiles for 'pclip' (e.g., 1 and 99)

    % choose bands
    bands = size(HSI_cells{1},3);
    wavelengths = linspace(waveStart_nm, waveEnd_nm, bands);
%     [~, idxNIR]   = min(abs(wavelengths - 800));
%     [~, idxRed]   = min(abs(wavelengths - 670));
%     [~, idxGreen] = min(abs(wavelengths - 550));
    
    [~, idxNIR]   = min(abs(wavelengths - 650));
    [~, idxRed]   = min(abs(wavelengths - 550));
    [~, idxGreen] = min(abs(wavelengths - 468));

    % stack all images’ selected bands to find global ranges
    allR = []; allG = []; allB = [];
    for k = 1:numel(HSI_cells)
        R = double(HSI_cells{k}(:,:,idxNIR));
        G = double(HSI_cells{k}(:,:,idxRed));
        B = double(HSI_cells{k}(:,:,idxGreen));
        allR = [allR; R(:)];
        allG = [allG; G(:)];
        allB = [allB; B(:)];
    end

    switch lower(method)
        case 'pclip'
            params.Rmin = prctile(allR, pLow);
            params.Gmin = prctile(allG, pLow);
            params.Bmin = prctile(allB, pLow);
            params.Rmax = prctile(allR, pHigh);
            params.Gmax = prctile(allG, pHigh);
            params.Bmax = prctile(allB, pHigh);
        case 'fixedmax'
            % single global [min,max] (useful if you know physical range)
            params.Rmin = 0;   params.Gmin = 0;   params.Bmin = 0;
            mx = max([max(allR), max(allG), max(allB)]);
            params.Rmax = mx;  params.Gmax = mx;  params.Bmax = mx;
        otherwise
            error('Unknown method');
    end

    % clamp guard
    epsv = 1e-12;
    params.Rmax = max(params.Rmax, params.Rmin+epsv);
    params.Gmax = max(params.Gmax, params.Gmin+epsv);
    params.Bmax = max(params.Bmax, params.Bmin+epsv);

    % keep band indices for reuse
    params.idxNIR = idxNIR;
    params.idxRed = idxRed;
    params.idxGreen = idxGreen;
end