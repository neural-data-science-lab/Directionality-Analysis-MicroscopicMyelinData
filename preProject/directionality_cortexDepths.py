import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import timeit
from scipy import ndimage
from scipy.ndimage import gaussian_filter

def directionality_cortexDepth(name_otsu, path, path_directionality, patch_size, nbr_cortexDepths=5, pixel=0.542):
    '''
    1. extract all valid patches in the sense that based on a binary mask only those orientation patches are valid in
    which the respective mask patch is not 0;
    2. rotate the orientations from directionality calculation in order to respect cortex curvature
    (Gradient filter over distance transform)
    3. sum over all patches with a certain cortex depth

    name_otsu:              file name of the threshold mask, test_C03_smooth3D_bg95_otsu.tif -> valid patches
    name_cortex:            file name of the cortex mask, test_C00_binMask_cortex.tif -> cortex curvature
    path:                   path to  files
    patch_directionality:   path to where the directionality calculation files lie
    patch_size:             size of square patch on which the orientation was computed
    pixel:                  1 pixel = 0.542 um

    returns dictionary d that contains the valid, corrected sums of orientations for specified cortex depths
    '''
    path_cortex = os.path.join(path, name_cortex)
    mask_cortex = io.imread(path_cortex)
    path_otsu = os.path.join(path, name_otsu)
    mask_otsu = io.imread(path_otsu)
    width = mask_cortex.shape[2]
    height = mask_cortex.shape[1]
    depth = mask_cortex.shape[0]

    # initialize the sum over the directionality
    file = path_directionality + str(0) + '_' + str(0) + '.csv'
    path_patch0 = os.path.join(path, file)
    patch0 = pd.read_csv(path_patch0)
    patch0.rename(columns={'Direction (�)': 'Direction'}, inplace=True)
    direction = pd.DataFrame(np.stack((patch0['Direction'], np.zeros(len(patch0['Direction']))), axis=1))

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
        orientations.append(sobel_smooth) # angles
        distances.append(dists) # distances to cortex
    layers = np.array([0,58.5,234.65,302.25,557.05])/pixel
    max_dist = 752.05/pixel

    d = {}  # dict for corrected valid directionality frequencies
    n = {}  # dict to sample for the nbr of patches per cortex depth
    for nbr in range(nbr_cortexDepths):
        d[str(nbr)] = np.copy(direction)
        n[str(nbr)] = 0

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
                if 255 in patch_otsu and cortexDepth <= max_dist: #cortexDepth > 0 and cortexDepth <= max_dist:
                    angle_cortex = orientations[k][int(j * patch_size + patch_size / 2),
                                                   int(i * patch_size + patch_size / 2)]
                    # get angle difference and rotate all orientations in patch
                    correction = 90 - angle_cortex
                    direction_corrected = patch['Direction'] - correction
                    # shift angles < -90 and > 90 degrees back into -90 to 90 range
                    patch_shifted = pd.concat([direction_corrected, patch['Slice_' + str(k + 1)]], axis=1)
                    patch_shifted['Direction'].loc[patch_shifted['Direction'] < -90] += 180
                    patch_shifted['Direction'].loc[patch_shifted['Direction'] > 90] -= 180
                    for row in range(len(patch_shifted)):
                        idx = (np.abs(d[str(key)][:, 0] - patch_shifted['Direction'][row])).argmin()
                        d[str(key)][idx, 1] += patch_shifted['Slice_' + str(k + 1)][row]
                    n[str(key)] += 1
    return d, n

# Plots
def plot_directionalityCorreted(data):
    labels = np.array(['I', 'II/III', 'IV', 'V', 'VI'])
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    for i in range(len(data)):
        ax.plot(data[str(i)][:, 0], data[str(i)][:, 1], label='layer ' + labels[i])
    ax.set_ylabel('Frequency of direction', fontsize=18)
    ax.set_xlabel('Directions in angluar degrees', fontsize=18)
    ax.set_title('Directionality of structures per cortex depth', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    fig.tight_layout()
    plt.show()

def plot_nbrPatchesInCortex(nbr):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.bar(np.arange(0, len(nbr.keys()), 1), nbr.values())
    ax.set_ylabel('# patches', fontsize=18)
    ax.set_xlabel('cortex depth', fontsize=18)
    ax.set_xticks(np.arange(0, len(nbr.keys()), 1))
    ax.set_xticklabels(nbr.keys(), fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    plt.show()


# main
name_cortex = 'test_C00_binMask_cortex.tif'
name_data = 'test_C03_smooth3D_bg95_sato.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/'
path_directionality = 'test_C03_smooth3D_bg95_sato_dice20/Sato_dice20_'
cortexDepths = 5
patch_size = 20

start = timeit.default_timer()
corrected, nbr = directionality_cortexDepth(name_cortex, path, path_directionality, patch_size=patch_size,
                                                   nbr_cortexDepths = cortexDepths)
plot_directionalityCorreted(corrected)
plot_nbrPatchesInCortex(nbr)
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in " + str(round(execution_time, 2)) + " seconds") #5060.08 seconds for 20x20

