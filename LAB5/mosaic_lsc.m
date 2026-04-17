% ============================================================
% LSC Mural Mosaicing Script
% EECE 5554 Lab 5
% Uses Harris corners + MATLAB feature matching
% ============================================================

clear all; close all;
addpath('/MATLAB Drive/LAB5/harris_files')
cd '/MATLAB Drive/LAB5/lsc_mural'

%% STEP 1: Load and resize images
% Resize to reduce memory and processing time
% Your images are very large (3072x4080) - resize to 25%
fprintf('Loading images...\n')
n_images = 5;
images = cell(1, n_images);

for i = 1:n_images
    fname = sprintf('LSC%d.jpg', i);
    I = imread(fname);
    % Resize to 25% for faster processing
    images{i} = imresize(I, 0.25);
    fprintf('Loaded %s: %dx%d\n', fname, size(images{i},1), size(images{i},2))
end

%% STEP 2: Detect Harris corners in each image
fprintf('\nDetecting Harris corners...\n')
points = cell(1, n_images);

for i = 1:n_images
    I_gray = double(rgb2gray(images{i}));
    [r, c, ~] = harris(I_gray, 500, 'tile', [2 3], 'thresh', 0);
    % Store as cornerPoints for MATLAB matching functions
    points{i} = [c, r];  % [x, y] format
    fprintf('Image %d: %d corners\n', i, size(points{i}, 1))
end

%% STEP 3: Extract features using MATLAB built-in
% We use SURF descriptors on Harris corner locations
% This combines Harris detection with robust SURF description
fprintf('\nExtracting feature descriptors...\n')
features = cell(1, n_images);
valid_points = cell(1, n_images);

for i = 1:n_images
    I_gray = rgb2gray(images{i});
    % Create corner points object from Harris locations
    hp = cornerPoints(points{i});
    % Extract SURF descriptors at Harris corner locations
    [features{i}, valid_points{i}] = extractFeatures(I_gray, hp);
    fprintf('Image %d: %d valid feature descriptors\n', i, size(features{i},1))
end

%% STEP 4: Match features between consecutive image pairs
fprintf('\nMatching features between consecutive images...\n')
tforms = cell(1, n_images);
tforms{1} = affinetform2d(eye(3));  % First image = reference (identity)

for i = 2:n_images
    % Match features between image i and image i-1
    indexPairs = matchFeatures(features{i-1}, features{i}, ...
        'Unique', true, 'MaxRatio', 0.7);

    fprintf('Images %d-%d: %d matches found\n', i-1, i, size(indexPairs,1))

    if size(indexPairs, 1) < 4
        fprintf('WARNING: Too few matches between images %d and %d!\n', i-1, i)
        fprintf('Try taking more overlapping photos.\n')
        tforms{i} = tforms{i-1};
        continue
    end

    % Get matched point locations
    matchedPts1 = valid_points{i-1}(indexPairs(:,1), :);
    matchedPts2 = valid_points{i}  (indexPairs(:,2), :);

    if i == 2
        figure;
        showMatchedFeatures(rgb2gray(images{i-1}), rgb2gray(images{i}), ...
            matchedPts1, matchedPts2, 'montage');
        title('Feature matches between LSC1 and LSC2')
        saveas(gcf, '/MATLAB Drive/LAB5/lsc_mural/matches_1_2.png')
    end

    % Estimate homography using RANSAC
    % RANSAC removes outlier matches automatically
    try
        tform = estgeotform2d(matchedPts2, matchedPts1, ...
            'affine', 'MaxNumTrials', 2000, ...
            'Confidence', 99, 'MaxDistance', 1.5);
        % Compose with previous transform to get global position
        tforms{i} = affinetform2d(tforms{i-1}.A * tform.A);
        fprintf('Images %d-%d: homography estimated successfully\n', i-1, i)
    catch ME
        fprintf('WARNING: Could not estimate homography: %s\n', ME.message)
        tforms{i} = tforms{i-1};
    end
end

%% STEP 5: Compute output panorama size
fprintf('\nComputing panorama dimensions...\n')

% Find the output limits of each transformed image
[height, width, ~] = size(images{1});
xlim = zeros(n_images, 2);
ylim = zeros(n_images, 2);

for i = 1:n_images
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms{i}, [1 width], [1 height]);
end

% Overall output size
xMin = min(xlim(:));  xMax = max(xlim(:));
yMin = min(ylim(:));  yMax = max(ylim(:));

panoramaWidth  = round(xMax - xMin);
panoramaHeight = round(yMax - yMin);
fprintf('Panorama size: %d x %d pixels\n', panoramaHeight, panoramaWidth)

% Limit size to avoid memory issues
maxDim = 8000;
if panoramaWidth > maxDim || panoramaHeight > maxDim
    scale = maxDim / max(panoramaWidth, panoramaHeight);
    panoramaWidth  = round(panoramaWidth  * scale);
    panoramaHeight = round(panoramaHeight * scale);
    fprintf('Resized to: %d x %d pixels\n', panoramaHeight, panoramaWidth)
end

%% STEP 6: Build the panorama
fprintf('\nBuilding panorama...\n')

% Create output view
xLimitsOut = [xMin, xMax];
yLimitsOut = [yMin, yMax];
panoramaView = imref2d([panoramaHeight panoramaWidth], xLimitsOut, yLimitsOut);

% Initialize blank panorama
panorama = zeros(panoramaHeight, panoramaWidth, 3, 'uint8');

% Warp and blend each image
blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

for i = 1:n_images
    fprintf('Warping image %d...\n', i)
    % Warp image into panorama coordinate system
    warpedImage = imwarp(images{i}, tforms{i}, 'OutputView', panoramaView);

    % Create mask for valid pixels
    mask = imwarp(true(size(images{i},1), size(images{i},2)), ...
        tforms{i}, 'OutputView', panoramaView);

    % Blend into panorama
    panorama = step(blender, panorama, warpedImage, mask);
end

%% STEP 7: Display and save
figure('Position', [100 100 1200 600]);
imshow(panorama);
title('LSC Mural Panoramic Mosaic', 'FontSize', 14);

saveas(gcf, '/MATLAB Drive/LAB5/lsc_mural/lsc_mosaic_final.png')
fprintf('\nMosaic saved: lsc_mosaic_final.png\n')
fprintf('Done!\n')