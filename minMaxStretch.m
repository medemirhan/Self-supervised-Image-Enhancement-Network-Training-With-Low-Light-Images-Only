function stretched_image = minMaxStretch(image, output_min, output_max)
% minMaxStretch Applies Min-Max contrast stretching to an image (2D or 3D).
%
%   stretched_image = minMaxStretch(image) stretches the input image to
%   the default range [0, 255].
%
%   stretched_image = minMaxStretch(image, output_min, output_max)
%   stretches the input image to the specified [output_min, output_max] range.
%
%   Args:
%       image: A 2D (grayscale) or 3D (e.g., RGB or hyperspectral)
%              numerical array. If 3D, stretching is applied band-wise.
%       output_min: (Optional) The desired minimum value of the output image.
%                   Default is 0.
%       output_max: (Optional) The desired maximum value of the output image.
%                   Default is 255.
%
%   Returns:
%       stretched_image: The contrast-stretched image, with the same
%                        dimensions as the input image. The data type will
%                        be double to preserve precision during stretching,
%                        unless the output range suggests an integer type
%                        (e.g. uint8 for [0 255]).

% --- Input Argument Handling ---
    if nargin < 1
        error('Error: Not enough input arguments. Please provide an image.');
    end
    if nargin < 2
        output_min = 0;
    end
    if nargin < 3
        output_max = 255;
    end

    if ~isnumeric(image)
        error('Error: Input image must be a numeric array.');
    end

    if output_min >= output_max
        error('Error: output_min must be less than output_max.');
    end

    % Convert image to double for precise calculations
    img_double = double(image);
    stretched_image = zeros(size(img_double), 'double'); % Initialize output

    % --- Perform Stretching ---
    if ndims(img_double) == 2 % Grayscale image
        min_val = min(img_double(:));
        max_val = max(img_double(:));

        if min_val == max_val % Handle constant image case
            stretched_image(:) = output_min; % Or output_max, or ((output_min+output_max)/2)
        else
            stretched_image = (img_double - min_val) ./ (max_val - min_val) * (output_max - output_min) + output_min;
        end

    elseif ndims(img_double) == 3 % Color image (RGB) or Hyperspectral (band-wise)
        num_bands = size(img_double, 3);
        for k = 1:num_bands
            band = img_double(:,:,k);
            min_val = min(band(:));
            max_val = max(band(:));

            if min_val == max_val % Handle constant band case
                stretched_image(:,:,k) = output_min;
            else
                stretched_image(:,:,k) = (band - min_val) ./ (max_val - min_val) * (output_max - output_min) + output_min;
            end
        end
    else
        error('Error: Image must be 2D or 3D.');
    end

    % --- Output Data Type Conversion (Optional) ---
    % If the target range is typical for an integer type, consider converting.
    % For example, if output_min is 0 and output_max is 255, convert to uint8.
    if output_min == 0 && output_max == 255
        stretched_image = uint8(stretched_image);
    elseif floor(output_min) == output_min && floor(output_max) == output_max && output_max <= intmax('uint16') && output_min >=0
        % Add more conditions if other integer types are desired
        % stretched_image = uint16(stretched_image);
    end

end