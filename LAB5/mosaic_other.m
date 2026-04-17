% ============================================================
% Third Subject Mosaicing Script - 15% and 50% Overlap
% EECE 5554 Lab 5
% ============================================================

clear all; close all;
addpath('/MATLAB Drive/LAB5/harris_files')

%% ============================================================
%% PART A: 50% OVERLAP
%% ============================================================
fprintf('========================================\n')
fprintf('PART A: 50 percent overlap set\n')
fprintf('========================================\n')

cd '/MATLAB Drive/LAB5/other_art/images_50'

n_50 = 5;
images_50 = cell(1, n_50);

% Load images
fprintf('Loading 50 percent overlap images...\n')
for i = 1:n_50
    fname = sprintf('50%d.jpg', i);
    I = imread(fname);
    images_50{i} = imresize(I, 0.25);
    fprintf('Loaded %s: %dx%d\n', fname, size(images_50{i},1), size(images_50{i},2))
end

% Harris corners
fprintf('\nDetecting Harris corners...\n')
points_50 = cell(1, n_50);
figure;
for i = 1:n_50
    I_gray = double(rgb2gray(images_50{i}));
    [r, c, ~] = harris(I_gray, 500, 'tile', [2 3], 'thresh', 0);
    points_50{i} = [c, r];
    subplot(2, 3, i);
    imshow(images_50{i});
    hold on;
    plot(c, r, 'r+', 'MarkerSize', 6, 'LineWidth', 1.5);
    title(sprintf('50%%-%d: %d corners', i, length(r)));
    hold off;
    fprintf('Image %d: %d corners\n', i, length(r))
end
sgtitle('Harris Corners - 50% Overlap Set');
saveas(gcf, '/MATLAB Drive/LAB5/other_art/harris_50.png')
fprintf('Saved: harris_50.png\n')

% Extract features
fprintf('\nExtracting features...\n')
features_50  = cell(1, n_50);
valid_pts_50 = cell(1, n_50);
for i = 1:n_50
    I_gray = rgb2gray(images_50{i});
    hp = cornerPoints(points_50{i});
    [features_50{i}, valid_pts_50{i}] = extractFeatures(I_gray, hp);
end

% Match and estimate transforms
fprintf('\nMatching features...\n')
tforms_50    = cell(1, n_50);
tforms_50{1} = affinetform2d(eye(3));

for i = 2:n_50
    indexPairs = matchFeatures(features_50{i-1}, features_50{i}, ...
        'Unique', true, 'MaxRatio', 0.7);
    fprintf('Images %d-%d: %d matches\n', i-1, i, size(indexPairs,1))

    if size(indexPairs,1) < 4
        fprintf('WARNING: Too few matches! Skipping transform.\n')
        tforms_50{i} = tforms_50{i-1};
        continue
    end

    matchedPts1 = valid_pts_50{i-1}(indexPairs(:,1), :);
    matchedPts2 = valid_pts_50{i}  (indexPairs(:,2), :);

    try
        tform = estgeotform2d(matchedPts2, matchedPts1, ...
            'affine', 'MaxNumTrials', 2000, ...
            'Confidence', 99, 'MaxDistance', 2.0);
        tforms_50{i} = affinetform2d(tforms_50{i-1}.A * tform.A);
        fprintf('Images %d-%d: transform OK\n', i-1, i)
    catch ME
        fprintf('WARNING: %s\n', ME.message)
        tforms_50{i} = tforms_50{i-1};
    end
end

% Build panorama
[height, width, ~] = size(images_50{1});
xlim = zeros(n_50,2); ylim = zeros(n_50,2);
for i = 1:n_50
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms_50{i}, [1 width], [1 height]);
end
xMin = min(xlim(:)); xMax = max(xlim(:));
yMin = min(ylim(:)); yMax = max(ylim(:));
pW = min(round(xMax-xMin), 8000);
pH = min(round(yMax-yMin), 8000);
fprintf('\nPanorama size: %d x %d\n', pH, pW)

pView    = imref2d([pH pW], [xMin xMax], [yMin yMax]);
panorama = zeros(pH, pW, 3, 'uint8');
blender  = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

for i = 1:n_50
    warpedImage = imwarp(images_50{i}, tforms_50{i}, 'OutputView', pView);
    mask = imwarp(true(size(images_50{i},1), size(images_50{i},2)), ...
        tforms_50{i}, 'OutputView', pView);
    panorama = step(blender, panorama, warpedImage, mask);
end

figure;
imshow(panorama);
title('Third Subject Mosaic - 50% Overlap', 'FontSize', 14);
saveas(gcf, '/MATLAB Drive/LAB5/other_art/mosaic_50_final.png')
fprintf('Saved: mosaic_50_final.png\n')

%% ============================================================
%% PART B: 15% OVERLAP
%% ============================================================
fprintf('\n========================================\n')
fprintf('PART B: 15 percent overlap set\n')
fprintf('========================================\n')

cd '/MATLAB Drive/LAB5/other_art/images_15'

n_15 = 4;
images_15 = cell(1, n_15);

% Load images
fprintf('Loading 15 percent overlap images...\n')
for i = 1:n_15
    fname = sprintf('15%d.jpg', i);
    I = imread(fname);
    images_15{i} = imresize(I, 0.25);
    fprintf('Loaded %s: %dx%d\n', fname, size(images_15{i},1), size(images_15{i},2))
end

% Harris corners
fprintf('\nDetecting Harris corners...\n')
points_15 = cell(1, n_15);
figure;
for i = 1:n_15
    I_gray = double(rgb2gray(images_15{i}));
    [r, c, ~] = harris(I_gray, 500, 'tile', [2 3], 'thresh', 0);
    points_15{i} = [c, r];
    subplot(2, 2, i);
    imshow(images_15{i});
    hold on;
    plot(c, r, 'r+', 'MarkerSize', 6, 'LineWidth', 1.5);
    title(sprintf('15%%-%d: %d corners', i, length(r)));
    hold off;
    fprintf('Image %d: %d corners\n', i, length(r))
end
sgtitle('Harris Corners - 15% Overlap Set');
saveas(gcf, '/MATLAB Drive/LAB5/other_art/harris_15.png')
fprintf('Saved: harris_15.png\n')

% Extract features
fprintf('\nExtracting features...\n')
features_15  = cell(1, n_15);
valid_pts_15 = cell(1, n_15);
for i = 1:n_15
    I_gray = rgb2gray(images_15{i});
    hp = cornerPoints(points_15{i});
    [features_15{i}, valid_pts_15{i}] = extractFeatures(I_gray, hp);
end

% Match and estimate transforms
fprintf('\nMatching features...\n')
tforms_15    = cell(1, n_15);
tforms_15{1} = affinetform2d(eye(3));

for i = 2:n_15
    indexPairs = matchFeatures(features_15{i-1}, features_15{i}, ...
        'Unique', true, 'MaxRatio', 0.7);
    fprintf('Images %d-%d: %d matches\n', i-1, i, size(indexPairs,1))

    if size(indexPairs,1) < 4
        fprintf('WARNING: Too few matches! Skipping transform.\n')
        tforms_15{i} = tforms_15{i-1};
        continue
    end

    matchedPts1 = valid_pts_15{i-1}(indexPairs(:,1), :);
    matchedPts2 = valid_pts_15{i}  (indexPairs(:,2), :);

    try
        tform = estgeotform2d(matchedPts2, matchedPts1, ...
            'affine', 'MaxNumTrials', 2000, ...
            'Confidence', 99, 'MaxDistance', 2.0);
        tforms_15{i} = affinetform2d(tforms_15{i-1}.A * tform.A);
        fprintf('Images %d-%d: transform OK\n', i-1, i)
    catch ME
        fprintf('WARNING: %s\n', ME.message)
        tforms_15{i} = tforms_15{i-1};
    end
end

% Build panorama
[height, width, ~] = size(images_15{1});
xlim = zeros(n_15,2); ylim = zeros(n_15,2);
for i = 1:n_15
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms_15{i}, [1 width], [1 height]);
end
xMin = min(xlim(:)); xMax = max(xlim(:));
yMin = min(ylim(:)); yMax = max(ylim(:));
pW = min(round(xMax-xMin), 8000);
pH = min(round(yMax-yMin), 8000);
fprintf('\nPanorama size: %d x %d\n', pH, pW)

pView    = imref2d([pH pW], [xMin xMax], [yMin yMax]);
panorama = zeros(pH, pW, 3, 'uint8');
blender  = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

for i = 1:n_15
    warpedImage = imwarp(images_15{i}, tforms_15{i}, 'OutputView', pView);
    mask = imwarp(true(size(images_15{i},1), size(images_15{i},2)), ...
        tforms_15{i}, 'OutputView', pView);
    panorama = step(blender, panorama, warpedImage, mask);
end

figure;
imshow(panorama);
title('Third Subject Mosaic - 15% Overlap', 'FontSize', 14);
saveas(gcf, '/MATLAB Drive/LAB5/other_art/mosaic_15_final.png')
fprintf('Saved: mosaic_15_final.png\n')
fprintf('\nAll done!\n')