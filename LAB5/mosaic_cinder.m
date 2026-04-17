% ============================================================
% Cinder Block Wall Mosaicing Script
% EECE 5554 Lab 5
% ============================================================

clear all; close all;
addpath('/MATLAB Drive/LAB5/harris_files')
cd '/MATLAB Drive/LAB5/cinder_block'

%% STEP 1: Load and resize images
fprintf('Loading images...\n')
n_images = 6;
images = cell(1, n_images);

for i = 1:n_images
    fname = sprintf('cinder%d.jpg', i);
    I = imread(fname);
    images{i} = imresize(I, 0.25);
    fprintf('Loaded %s: %dx%d\n', fname, size(images{i},1), size(images{i},2))
end

%% STEP 2: Detect Harris corners
fprintf('\nDetecting Harris corners...\n')
points = cell(1, n_images);

for i = 1:n_images
    I_gray = double(rgb2gray(images{i}));
    [r, c, ~] = harris(I_gray, 500, 'tile', [2 3], 'thresh', 0);
    points{i} = [c, r];
    fprintf('Image %d: %d corners\n', i, size(points{i}, 1))
end

%% STEP 3: Save Harris corners plot for report
figure;
for i = 1:n_images
    subplot(2, 3, i);
    imshow(images{i});
    hold on;
    plot(points{i}(:,1), points{i}(:,2), 'r+', ...
        'MarkerSize', 6, 'LineWidth', 1.5);
    title(sprintf('cinder%d: %d corners', i, size(points{i},1)));
    hold off;
end
sgtitle('Harris Corners on Cinder Block Images');
saveas(gcf, '/MATLAB Drive/LAB5/cinder_block/harris_all_cinder.png')
fprintf('Saved: harris_all_cinder.png\n')

%% STEP 4: Extract features
fprintf('\nExtracting feature descriptors...\n')
features  = cell(1, n_images);
valid_pts = cell(1, n_images);

for i = 1:n_images
    I_gray = rgb2gray(images{i});
    hp = cornerPoints(points{i});
    [features{i}, valid_pts{i}] = extractFeatures(I_gray, hp);
end

%% STEP 5: Match and estimate transforms
fprintf('\nMatching features...\n')
tforms    = cell(1, n_images);
tforms{1} = affinetform2d(eye(3));

for i = 2:n_images
    indexPairs = matchFeatures(features{i-1}, features{i}, ...
        'Unique', true, 'MaxRatio', 0.7);

    fprintf('Images %d-%d: %d matches found\n', i-1, i, size(indexPairs,1))

    if size(indexPairs, 1) < 4
        fprintf('WARNING: Too few matches between images %d and %d!\n', i-1, i)
        tforms{i} = tforms{i-1};
        continue
    end

    matchedPts1 = valid_pts{i-1}(indexPairs(:,1), :);
    matchedPts2 = valid_pts{i}  (indexPairs(:,2), :);

    % Visualize matches for first pair
    if i == 2
        figure;
        showMatchedFeatures(rgb2gray(images{i-1}), rgb2gray(images{i}), ...
            matchedPts1, matchedPts2, 'montage');
        title('Feature Matches: cinder1 vs cinder2')
        saveas(gcf, '/MATLAB Drive/LAB5/cinder_block/matches_cinder_1_2.png')
    end

    try
        tform = estgeotform2d(matchedPts2, matchedPts1, ...
            'affine', 'MaxNumTrials', 2000, ...
            'Confidence', 99, 'MaxDistance', 2.0);
        tforms{i} = affinetform2d(tforms{i-1}.A * tform.A);
        fprintf('Images %d-%d: transform estimated successfully\n', i-1, i)
    catch ME
        fprintf('WARNING: %s\n', ME.message)
        tforms{i} = tforms{i-1};
    end
end

%% STEP 6: Build panorama
fprintf('\nBuilding panorama...\n')
[height, width, ~] = size(images{1});
xlim = zeros(n_images, 2);
ylim = zeros(n_images, 2);

for i = 1:n_images
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms{i}, [1 width], [1 height]);
end

xMin = min(xlim(:)); xMax = max(xlim(:));
yMin = min(ylim(:)); yMax = max(ylim(:));

panoramaWidth  = round(xMax - xMin);
panoramaHeight = round(yMax - yMin);
fprintf('Panorama size: %d x %d pixels\n', panoramaHeight, panoramaWidth)

% Limit size
maxDim = 8000;
if panoramaWidth > maxDim || panoramaHeight > maxDim
    scale = maxDim / max(panoramaWidth, panoramaHeight);
    panoramaWidth  = round(panoramaWidth  * scale);
    panoramaHeight = round(panoramaHeight * scale);
end

panoramaView = imref2d([panoramaHeight panoramaWidth], ...
    [xMin xMax], [yMin yMax]);
panorama = zeros(panoramaHeight, panoramaWidth, 3, 'uint8');

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

for i = 1:n_images
    fprintf('Warping image %d...\n', i)
    warpedImage = imwarp(images{i}, tforms{i}, 'OutputView', panoramaView);
    mask = imwarp(true(size(images{i},1), size(images{i},2)), ...
        tforms{i}, 'OutputView', panoramaView);
    panorama = step(blender, panorama, warpedImage, mask);
end

%% STEP 7: Display and save
figure;
imshow(panorama);
title('Cinder Block Wall Mosaic', 'FontSize', 14);
saveas(gcf, '/MATLAB Drive/LAB5/cinder_block/cinder_mosaic_final.png')
fprintf('Saved: cinder_mosaic_final.png\n')
fprintf('Done!\n')