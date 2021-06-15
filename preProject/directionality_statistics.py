import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import timeit
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import circmean, circvar, circstd, mode
from skimage.io import imsave


def rotate_directionality(name_otsu, name_cortex, path, path_directionality, patch_size=80, statistics=True, z_slice=0):
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

    orientations = []
    for z in range(depth):
        d[str(z)] = np.copy(direction)
        uncorrected[str(z)] = np.copy(direction)
        # calculate gradient filter over distance transform per z-slice
        slice = mask_cortex[z]
        distances = ndimage.distance_transform_edt(slice, return_distances=True)
        sx = ndimage.sobel(distances, axis=0, mode='nearest')
        sy = ndimage.sobel(distances, axis=1, mode='nearest')
        sobel = np.arctan2(sy, sx) * 180 / np.pi
        # smooth sobel
        sobel_smooth = gaussian_filter(sobel, sigma=2)
        orientations.append(sobel_smooth)

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
                if 255 in patch_otsu:
                    uncorrected[str(k)][:, 1] += patch['Slice_' + str(k + 1)]
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
                        idx = (np.abs(d[str(k)][:, 0] - patch_shifted['Direction'][row])).argmin()  # nearest value in directions
                        d[str(k)][idx, 1] += patch_shifted['Slice_' + str(k + 1)][row]
                        summary[idx, 1] = patch_shifted['Slice_' + str(k + 1)][row]

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
                        # round(circmean(summary_u[:, 0] * summary_u[:, 1], high=np.pi, low=-np.pi), 5) or round(mode(summary[:, 0] * summary[:, 1])[0][0], 5)
                        # s[str(k)].append(stats)
                        s.append(stats)
                        u.append(stats_u)
    return d, uncorrected, s, u


# Plots
def plot_directionalityCorreted(data):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    for i in range(len(data)):
        ax.plot(data[str(i)][:, 0], data[str(i)][:, 1], label='slice ' + str(i))
    ax.set_ylabel('Frequency of direction', fontsize=18)
    ax.set_xlabel('Directions in angluar degrees', fontsize=18)
    ax.set_title('Directionality of structures in degrees', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
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
    # ax.plot(np.array(X), np.array(Y), 'ro', markersize = 2)
    plt.show()


# main
name_otsu = 'test_C03_smooth3D_bg95_otsu.tif'
name_cortex = 'test_C00_binMask_cortex.tif'
name_data = 'test_C03_smooth3D_bg95_sato.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/'
path_directionality = 'test_C03_smooth3D_bg95_sato_dice20/Sato_dice20_'
patch_size = 20

start = timeit.default_timer()
corrected, uncorrected, stats_corr, stats_uncorr = rotate_directionality(name_otsu, name_cortex, path, path_directionality,
                                                                         patch_size=20, statistics=True, z_slice=0)
plot_directionalityCorreted(corrected)
plot_directionalityCorreted(uncorrected)
plot_directionalityStatistics(name_data, path, stats_corr, slice=0)
plot_directionalityStatistics(name_data, path, stats_uncorr, slice=0)
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in " + str(round(execution_time, 2)) + " seconds")
# 3.61 min vs 3.93 min (directionality_analysis.py):  before uncorrected version

