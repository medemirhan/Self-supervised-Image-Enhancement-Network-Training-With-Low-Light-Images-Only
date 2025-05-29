function out = pctStretch(in, pLow, pHigh, doScale)
% pctStretch  Clip image data to [pLow,pHigh] percentiles, optional scale
%
%   out = pctStretch(in, pLow, pHigh)
%   out = pctStretch(in, pLow, pHigh, doScale)
%
%   Inputs:
%     in      – 2D array
%     pLow    – lower percentile (e.g. 2)
%     pHigh   – upper percentile (e.g. 98)
%     doScale – (optional) logical; if true (default), scale clipped
%               values to [0,1]. If false, return clipped values unchanged.
%
%   Output:
%     out     – clipped (and optionally scaled) image

    if nargin < 4
        doScale = true;
    end

    lo = prctile(in(:), pLow);
    hi = prctile(in(:), pHigh);

    % Clip to [lo, hi]
    clipped = min(max(in, lo), hi);

    if doScale
        % Scale linearly to [0,1]
        out = (clipped - lo) ./ (hi - lo);
    else
        % Return clipped data without scaling
        out = clipped;
    end
end