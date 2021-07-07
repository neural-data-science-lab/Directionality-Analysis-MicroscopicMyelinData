import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('side', type=str)
parser.add_argument('patch_size', type=int)
parser.add_argument('slice', type=int)
args = parser.parse_args()

def statistics(name_cortex, path, path_directionality, patch_size, slice=0):
    '''
    function to obtain a statistics from the directionality analysis

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
    data.rename(columns={'Direction (�)': 'Direction'}, inplace=True)
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
            patch.rename(columns={'Direction (�)': 'Direction'}, inplace=True)
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
    #plt.savefig(save_path + 'Statistics_' + slice + '.png', dpi=200)

#side = 'Left'
#patch_size = 80
#slice = 0
name_cortex = args.side + '_cortex.tif'
name_data = args.side + '_smooth2_bg95_frangi2.tif'
path = '/media/muellerg/Data SSD/Gesine/Data/'
folder_directionality = args.side + '_frangi_' + str(args.patch_size)+'/'
name_directionality = args.side + str(args.patch_size) + '_'
path_directionality = folder_directionality + name_directionality
cortexDepths = 5
save_path = path+folder_directionality

s = statistics(name_cortex, path, path_directionality, args.patch_size, slice=args.slice)
plot_Statistics(name_data, path, s, args.patch_size, save_path, slice=args.slice)