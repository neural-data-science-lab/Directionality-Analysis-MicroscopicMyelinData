import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import timeit
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import circmean, circvar, circstd


def rotate_directionality(name_otsu, name_cortex, path, path_directionality, patch_size, nbr_cortexDepths=5, pixel=0.542,
                          statistics=True, z_slice=0):
    '''
    1. extract all valid patches in the sense that based on a binary mask only those orientation patches are valid in
    which the respective mask patch is not 0;
    2. rotate the orientations from directionality calculation in order to respect cortex curvature
    (Gradient filter over distance transform)
    3. sum over all patches in one z-slice
    4. Obtain little statistics for every valid patch for specific z-slice (can be adapted to all z-slices)

    name_otsu:              file name of the threshold mask, test_C03_smooth3D_bg95_otsu.tif
    name_cortex:            file name of the cortex mask, test_C00_binMask_cortex.tif
    path:                   path to  files
    patch_directionality:   path to where the directionality calculation files lie
    patch_size:             size of square patch on which the orientation was computed

    returns dictionary d that contains the valid, corrected sums of orientations for all z-slices
    returns dictionary s that contains the statistics for every valid patch and every z-slice
    '''

    path_otsu = os.path.join(path, name_otsu)
    mask_otsu = io.imread(path_otsu)
    path_cortex = os.path.join(path, name_cortex)
    mask_cortex = io.imread(path_cortex)

    # dimensionality of data
    width = mask_otsu.shape[2]
    height = mask_otsu.shape[1]
    depth = mask_otsu.shape[0]

    # initialize the sum over the directionality
    file = path_directionality + str(0) + '_' + str(0) + '.csv'
    path_patch0 = os.path.join(path, file)
    patch0 = pd.read_csv(path_patch0)
    patch0.rename(columns={'Direction (�)': 'Direction'}, inplace=True)
    direction = pd.DataFrame(np.stack((patch0['Direction'], np.zeros(len(patch0['Direction']))), axis=1))

    d = {}  # dict for corrected valid directionality frequencies
    uncorrected = {}  # dict for uncorrected valid directionality frequencies
    n = {}  # dict to sample for the nbr of patches per cortex depth
    for nbr in range(nbr_cortexDepths):
        d[str(nbr)] = np.copy(direction)
        uncorrected[str(nbr)] = np.copy(direction)
        n[str(nbr)] = 0

    distances = []
    orientations = []
    for z in range(depth):
        slice = mask_cortex[z]
        dists = ndimage.distance_transform_edt(slice, return_distances=True)
        sx = ndimage.sobel(dists, axis=0, mode='nearest')
        sy = ndimage.sobel(dists, axis=1, mode='nearest')
        sobel = np.arctan2(sy, sx) * 180 / np.pi
        # smooth sobel
        sobel_smooth = gaussian_filter(sobel, sigma=2)
        orientations.append(sobel_smooth)  # angles
        distances.append(dists)  # distances to cortex
    layers = np.array([0, 58.5, 234.65, 302.25, 557.05]) / pixel
    max_dist = 752.05 / pixel

    if statistics == True:
        # s = {}  # dict for the statistics if statistics is required for all z-slices
        # for z in range(depth): s[str(z)] = []
        s = []  # for the corrected version
        u = []  # for the uncorrected version

    for i in range(int(width / patch_size)):
        for j in range(int(height / patch_size)):
            filename = path_directionality + str(i) + '_' + str(j) + '.csv'
            path_patch = os.path.join(path, filename)
            patch = pd.read_csv(path_patch)
            patch.rename(columns={'Direction (�)': 'Direction'}, inplace=True)

            for k in range(depth):
                patch_otsu = mask_otsu[k, j * patch_size:j * patch_size + patch_size,
                             i * patch_size:i * patch_size + patch_size]
                cortexDepth = distances[k][int(j * patch_size + patch_size / 2), int(i * patch_size + patch_size / 2)]
                key = np.digitize(cortexDepth, layers, right=False) - 1
                if 255 in patch_otsu and cortexDepth <= max_dist:
                    uncorrected[str(key)][:, 1] += patch['Slice_' + str(k + 1)]
                    # rotate orientations according to cortex curvature
                    angle_cortex = orientations[k][int(j * patch_size + patch_size / 2),
                                                   int(i * patch_size + patch_size / 2)]  # middle point of box
                    # get angle difference and rotate all orientations in patch
                    correction = 90 - angle_cortex
                    direction_corrected = patch['Direction'] - correction
                    # shift angles < -90 and > 90 degrees back into -90 to 90 range
                    patch_shifted = pd.concat([direction_corrected, patch['Slice_' + str(k + 1)]], axis=1)
                    patch_shifted['Direction'].loc[patch_shifted['Direction'] < -90] += 180
                    patch_shifted['Direction'].loc[patch_shifted['Direction'] > 90] -= 180
                    summary = np.copy(direction)
                    summary_u = np.stack((patch0['Direction'], patch['Slice_' + str(k + 1)]), axis=1)
                    for row in range(len(patch_shifted)):
                        idx = (np.abs(d[str(key)][:, 0] - patch_shifted['Direction'][row])).argmin()  # nearest value in directions
                        d[str(key)][idx, 1] += patch_shifted['Slice_' + str(k + 1)][row]
                        summary[idx, 1] = patch_shifted['Slice_' + str(k + 1)][row]
                    n[str(key)] += 1

                    if (statistics == True) and (k == z_slice):
                        summary[:, 0] = np.radians(summary[:, 0] / math.pi)
                        summary_u[:, 0] = np.radians(summary_u[:, 0] / math.pi)
                        stats = np.array(
                            [i, j, correction, summary[summary[:,1].argmax(),0],
                             round(circvar(summary[:, 0] * summary[:, 1], high=np.pi, low=-np.pi), 5),
                             round(circstd(summary[:, 0] * summary[:, 1], high=np.pi, low=-np.pi), 5)])
                        stats_u = np.array(
                            [i, j, 0, summary_u[summary_u[:,1].argmax(),0],
                             round(circvar(summary_u[:, 0] * summary_u[:, 1], high=np.pi, low=-np.pi), 5),
                             round(circstd(summary_u[:, 0] * summary_u[:, 1], high=np.pi, low=-np.pi), 5)])
                        # s[str(k)].append(stats)
                        s.append(stats)
                        u.append(stats_u)
    return d, uncorrected, s, u, n


# Plots
def plot_directionalityCorreted(data, nbr, normalize = True):
    labels = np.array(['I', 'II/III', 'IV', 'V', 'VI'])
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    s = sum(nbr.values())
    for i in range(len(data)):
        if normalize:
            title = 'Normalized directionality analysis'
            freq = data[str(i)][:, 1] / (nbr[str(i)])
        else:
            freq = data[str(i)][:, 1]
            title = 'Directionality analysis'
        ax.plot(data[str(i)][:, 0],freq, label='layer ' + labels[i])
    ax.set_ylabel('Frequency of direction', fontsize=18)
    ax.set_xlabel('Directions in angluar degrees', fontsize=18)
    ax.set_title(title, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_directionalityStatistics(name_data, path, statistics, slice=0):
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
    plt.show()

def plot_nbrPatchesInCortex(nbr):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.bar(np.arange(0, len(nbr.keys()), 1), nbr.values())
    ax.set_ylabel('# patches', fontsize=18)
    ax.set_xlabel('cortex depth', fontsize=18)
    ax.set_xticks(np.arange(0, len(nbr.keys()), 1))
    ax.set_xticklabels(np.array(['I', 'II/III', 'IV', 'V', 'VI']), fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    plt.show()


# main
name_otsu = 'test_C03_smooth3D_bg95_otsu.tif'
name_cortex = 'test_C00_binMask_cortex.tif'
name_data = 'test_C03_smooth3D_bg95_sato.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/'
path_directionality = 'test_C03_smooth3D_bg95_sato-dice80/directionalitySato_smoothBG80_'
patch_size = 80

start = timeit.default_timer()
corrected, uncorrected, stats_corr, stats_uncorr, nbr = \
    rotate_directionality(name_otsu, name_cortex, path, path_directionality, patch_size, nbr_cortexDepths=5,
                          pixel=0.542,statistics=True, z_slice=0)
plot_directionalityCorreted(corrected, nbr, normalize = True)
plot_directionalityCorreted(uncorrected, nbr, normalize = True)
plot_directionalityStatistics(name_data, path, stats_corr, slice=0)
plot_directionalityStatistics(name_data, path, stats_uncorr, slice=0)
plot_nbrPatchesInCortex(nbr)
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in " + str(round(execution_time, 2)) + " seconds")
# 3.61 min vs 3.93 min (directionality_analysis.py):  before uncorrected version

