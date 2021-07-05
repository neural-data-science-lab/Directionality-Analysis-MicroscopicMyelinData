import math
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import timeit
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import json
import psutil


def directionality_cortexDepth(name_cortex, path, path_directionality, patch_size, nbr_cortexDepths=5, pixel=0.542):
    '''
    1. extract all valid patches in the sense that based on a binary mask only those orientation patches are valid in
    which the respective mask patch is not 0;
    2. rotate the orientations from directionality calculation in order to respect cortex curvature
    (Gradient filter over distance transform)
    3. sum over all patches with a certain cortex depth

    name_cortex:            file name of the cortex mask, test_C00_binMask_cortex.tif -> cortex curvature
    path:                   path to  files
    patch_directionality:   path to where the directionality calculation files lie
    patch_size:             size of square patch on which the orientation was computed
    pixel:                  1 pixel = 0.542 um

    returns dictionary d that contains the valid, corrected sums of orientations for specified cortex depths
    '''
    path_cortex = os.path.join(path, name_cortex)
    mask_cortex = io.imread(path_cortex)
    width = mask_cortex.shape[2]
    height = mask_cortex.shape[1]
    depth = mask_cortex.shape[0]

    # initialize the sum over the directionality
    file = path_directionality + str(0) + '_' + str(0) + '.csv'
    path_patch0 = os.path.join(path, file)
    patch0 = pd.read_csv(path_patch0)
    patch0.rename(columns={'Direction (°)': 'Direction'}, inplace=True)
    direction = patch0['Direction']

    n = pd.DataFrame(np.zeros((1, nbr_cortexDepths)))  # pd to sample for the nbr of patches per cortex depth
    d = pd.DataFrame(np.column_stack((np.copy(direction), np.zeros(
        (len(direction), nbr_cortexDepths)))))  # pd for corrected valid directionality frequencies

    batch_size = 50
    for batch in range(0, int(depth / batch_size)):
        orientations = []
        dists3D = ndimage.distance_transform_edt(mask_cortex[batch * batch_size:batch * batch_size + batch_size, :, :],
                                                 sampling=[8, 1, 1], return_distances=True)

        for z in range(batch_size):
            sx = ndimage.sobel(dists3D[z], axis=0, mode='nearest')
            sy = ndimage.sobel(dists3D[z], axis=1, mode='nearest')
            sobel = np.arctan2(sy, sx) * 180 / np.pi
            sobel_smooth = gaussian_filter(sobel, sigma=2)
            orientations.append(sobel_smooth)  # angles
        layers = np.array([0, 60, 235, 300, 560]) / pixel
        max_dist = 750 / pixel

        for i in range(int(width / patch_size)):
            for j in range(int(height / patch_size)):
                filename = path_directionality + str(i) + '_' + str(j) + '.csv'
                path_patch = os.path.join(path, filename)
                patch = pd.read_csv(path_patch)
                patch.rename(columns={'Direction (°)': 'Direction'}, inplace=True)

                x = np.arange(batch * batch_size, batch * batch_size + batch_size, 1)
                for k, v in enumerate(x):
                    cortexDepth = dists3D[k][int(j * patch_size + patch_size / 2), int(i * patch_size + patch_size / 2)]
                    key = np.digitize(cortexDepth, layers, right=False)
                    if cortexDepth <= max_dist and np.isnan(np.min(patch['Slice_' + str(v + 1)])) == False:
                        angle_cortex = orientations[k][int(j * patch_size + patch_size / 2),
                                                       int(i * patch_size + patch_size / 2)]
                        # get angle difference and rotate all orientations in patch
                        correction = 90 - angle_cortex
                        direction_corrected = patch['Direction'] - correction
                        # shift angles < -90 and > 90 degrees back into -90 to 90 range
                        patch_shifted = pd.concat([direction_corrected, patch['Slice_' + str(v + 1)]], axis=1)
                        patch_shifted['Direction'].loc[patch_shifted['Direction'] < -90] += 180
                        patch_shifted['Direction'].loc[patch_shifted['Direction'] > 90] -= 180
                        for row in range(len(patch_shifted)):
                            idx = (np.abs(d[0] - patch_shifted['Direction'][row])).argmin()
                            d[key][idx] += patch_shifted['Slice_' + str(v + 1)][row]
                        n[key - 1] += 1
    return d, n


# Plots
def plot_directionalityCorreted(data, nbr, save_path, normalize=False):
    labels = nbr.keys()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    for i, l in enumerate(labels):
        if normalize:
            title = 'Normalized directionality analysis'
            freq = data[l] / (nbr[l])
            name = 'Directionality_norm.png'
        else:
            freq = data[l]
            title = 'Directionality analysis'
            name = 'Directionality.png'
        ax.plot(data['Direction'], freq, label='layer ' + l)
    ax.set_ylabel('Frequency of direction', fontsize=18)
    ax.set_xlabel('Directions in angluar degrees', fontsize=18)
    ax.set_title(title, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    fig.tight_layout()
    # save figure
    plt.savefig(save_path + name, dpi=200)


def plot_nbrPatchesInCortex(nbr, save_path):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.bar(np.arange(0, len(nbr.keys()), 1), nbr.values())
    ax.set_ylabel('# patches', fontsize=18)
    ax.set_xlabel('cortex depth', fontsize=18)
    ax.set_xticks(np.arange(0, len(nbr.keys()), 1))
    ax.set_xticklabels(np.array(['I', 'II/III', 'IV', 'V', 'VI']), fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    # save figure
    plt.savefig(save_path + 'nbrPatches.png', dpi=200)


# Statistics
def statistics(name_cortex, path, path_directionality, patch_size, slice=0):
    '''
    function to obtain a statistics from the directionality analysis

    name_cortex:            file name of the cortex mask, test_C00_binMask_cortex.tif
    path:                   path to  files
    patch_directionality:   path to where the directionality calculation files lie
    patch_size:             size of square patch on which the orientation was computed
    slice:                  2D data; z-slice which is used for statistics

    output of the function gives the i,j position of the respective patch, the angle of the correction towards the
    cortex normals and mode
    '''
    path_cortex = os.path.join(path, name_cortex)
    mask_cortex = io.imread(path_cortex)[slice]
    width = mask_cortex.shape[1]
    height = mask_cortex.shape[0]
    file = path_directionality + str(0) + '_' + str(0) + '.csv'
    path_data = os.path.join(path, file)
    data = pd.read_csv(path_data)
    data.rename(columns={'Direction (°)': 'Direction'}, inplace=True)
    direction = pd.DataFrame(np.stack((data['Direction'], np.zeros(len(data['Direction']))), axis=1))
    distances = ndimage.distance_transform_edt(mask_cortex, return_distances=True)
    sx = ndimage.sobel(distances, axis=0, mode='nearest')
    sy = ndimage.sobel(distances, axis=1, mode='nearest')
    sobel = np.arctan2(sy, sx) * 180 / np.pi
    orientations = gaussian_filter(sobel, sigma=2)
    max_dist = 752.05 / 0.542

    d = []
    for i in range(int(width / patch_size)):
        for j in range(int(height / patch_size)):
            filename = path_directionality + str(i) + '_' + str(j) + '.csv'
            path_patch = os.path.join(path, filename)
            patch = pd.read_csv(path_patch)
            patch.rename(columns={'Direction (°)': 'Direction'}, inplace=True)
            patch_otsu = mask_otsu[j * patch_size:j * patch_size + patch_size,
                         i * patch_size:i * patch_size + patch_size]
            cortexDepth = distances[int(j * patch_size + patch_size / 2), int(i * patch_size + patch_size / 2)]
            if cortexDepth <= max_dist and np.isnan(np.min(patch['Slice_' + str(slice + 1)])) == False:
                angle_cortex = orientations[int(j * patch_size + patch_size / 2), int(i * patch_size + patch_size / 2)]
                correction = 90 - angle_cortex
                direction_corrected = patch['Direction'] - correction
                patch_shifted = pd.concat([direction_corrected, patch['Slice_' + str(slice + 1)]], axis=1)
                patch_shifted['Direction'].loc[patch_shifted['Direction'] < -90] += 180
                patch_shifted['Direction'].loc[patch_shifted['Direction'] > 90] -= 180
                summary = np.copy(direction)
                for row in range(len(patch_shifted)):
                    idx = (np.abs(summary[:, 0] - patch_shifted['Direction'][row])).argmin()
                    summary[idx, 1] = patch_shifted['Slice_' + str(slice + 1)][row]
                summary[:, 0] = np.radians(summary[:, 0] / math.pi)
                stats = np.array([i, j, correction, summary[summary[:, 1].argmax(), 0]])
                d.append(stats)
    return d


def plot_Statistics(name_data, path, statistics, patch_size, save_path, slice=0):
    path_data = os.path.join(path, name_data)
    data = io.imread(path_data)[slice]
    stats = pd.DataFrame(statistics)
    X = stats[0] * patch_size + patch_size / 2
    Y = stats[1] * patch_size + patch_size / 2
    angles = stats[2] + np.degrees(stats[3] * math.pi)
    U = np.cos(angles * np.pi / 180)
    V = np.sin(angles * np.pi / 180)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(data, cmap="gray")
    ax.quiver(X, Y, U, V, color='red', units='xy')
    # save figure
    plt.savefig(save_path + 'Statistics_' + slice + '.png', dpi=200)


# main
side = 'Left'
patch_size = 40
name_cortex = side + '_cortex.tif'
name_data = side + '_smooth2_bg95_frangi2.tif'
name_otsu = side + '_smooth2_bg95_otsu.tif'
path = '/media/muellerg/Data SSD/Gesine/Data/'
folder_directionality = side + '_frangi_' + str(patch_size) + '/'
name_directionality = side + str(patch_size) + '_'
path_directionality = folder_directionality + name_directionality
cortexDepths = 5
save_path = path + folder_directionality

start = timeit.default_timer()
corrected, nbr = directionality_cortexDepth(name_otsu, name_cortex, path, path_directionality, patch_size,
                                            nbr_cortexDepths=5, pixel=0.542)
pickle.dump(corrected, open(path + '/' + folder_directionality + 'corrected-Otsu.pkl', 'wb'))
json.dump(nbr, open(path + '/' + folder_directionality + 'nbr-Otsu.json', 'w'))

# plot_directionalityCorreted(corrected, nbr, save_path, normalize = True)
# plot_directionalityCorreted(corrected, nbr, save_path, normalize = False)
# plot_nbrPatchesInCortex(nbr, save_path)
# stats = statistics(name_otsu, name_cortex, path, path_directionality, patch_size, slice = 0)
# plot_Statistics(name_data, path, stats, patch_size, save_path, slice=0)
stop = timeit.default_timer()
execution_time = stop - start
print('Program Executed in ' + str(round(execution_time, 2)) + ' seconds')  # 1642.25 seconds for 20x20