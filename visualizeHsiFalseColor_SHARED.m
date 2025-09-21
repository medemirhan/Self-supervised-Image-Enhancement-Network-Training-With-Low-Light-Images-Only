function falseColorImage = visualizeHsiFalseColor_SHARED(HSI, params, gammaVal)
% HSI: one cube
% params: from computeFalseColorDisplayParams
% gammaVal: e.g., 0.8 (optional). Use 1.0 to disable.

    if nargin < 3 || isempty(gammaVal), gammaVal = 1.0; end

    R = double(HSI(:,:,params.idxNIR));
    G = double(HSI(:,:,params.idxRed));
    B = double(HSI(:,:,params.idxGreen));

    % global linear scale per channel (shared across images)
    R = (R - params.Rmin) / (params.Rmax - params.Rmin);
    G = (G - params.Gmin) / (params.Gmax - params.Gmin);
    B = (B - params.Bmin) / (params.Bmax - params.Bmin);

    % clamp
    R = min(max(R,0),1);
    G = min(max(G,0),1);
    B = min(max(B,0),1);

    % optional gamma to enhance perceptual separation
    if gammaVal ~= 1.0
        R = R .^ (1/gammaVal);
        G = G .^ (1/gammaVal);
        B = B .^ (1/gammaVal);
    end

    falseColorImage = cat(3, R, G, B);
end