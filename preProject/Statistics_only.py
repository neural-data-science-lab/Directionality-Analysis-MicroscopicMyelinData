import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import argparse
import matplotlib.cm as cm
from matplotlib.colors import Normalize

'''parser = argparse.ArgumentParser()
parser.add_argument('side', type=str)
parser.add_argument('patch_size', type=int)
parser.add_argument('slice', type=int)
args = parser.parse_args()'''

def statistics(name_cortex, name_otsu, path, path_directionality, patch_size, slice=0):
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
    path_otsu = os.path.join(path, name_otsu)
    mask_otsu = io.imread(path_otsu)[slice]
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
            patch_otsu = mask_otsu[j * patch_size:j * patch_size + patch_size, i * patch_size:i * patch_size + patch_size]
            cortexDepth = distances[int(j * patch_size + patch_size / 2), int(i * patch_size + patch_size / 2)]
            if 255 in patch_otsu and cortexDepth <= max_dist and np.isnan(np.min(patch['Slice_' + str(slice + 1)])) == False:
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
    colors = np.array(np.arctan2(U,V)) #ToDo: vllt U,V tauschen -> einmal ausprobieren
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.viridis
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(data, cmap="gray")
    ax.quiver(X, Y, U, V, color=colormap(norm(colors)), units='xy', linewidths=12)
    norm.autoscale(np.array(angles))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, fraction=0.035)
    #ax.quiver(X, Y, U, V, color='red', units='xy')
    # save figure
    #plt.savefig(save_path + 'Statistics_' + str(slice) + '.png', dpi=200)

#side = 'Left'

'''name_cortex = args.side + '_cortex.tif'
name_otsu = args.side + '_smooth2_bg95_otsu.tif'
name_data = args.side + '_smooth2_bg95_frangi2.tif'
path = '/media/muellerg/Data SSD/Gesine/Data/'
folder_directionality = args.side + '_frangi_' + str(args.patch_size)+'/'
name_directionality = args.side + str(args.patch_size) + '_'
path_directionality = folder_directionality + name_directionality
cortexDepths = 5
save_path = path+folder_directionality
s = statistics(name_cortex, name_otsu, path, path_directionality, args.patch_size, slice=args.slice)
plot_Statistics(name_data, path, s, args.patch_size, save_path, slice=args.slice)'''

name_cortex = 'test_C00_binMask_cortex.tif'
name_data = 'test_C03_smooth3D_bg95_frangi.tif'
name_otsu = 'test_C03_smooth3D_bg95_otsu.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/test/'
folder_directionality = 'dir_92/'
name_directionality = 'Left92'
path_directionality = folder_directionality + name_directionality
save_path = path + folder_directionality
patch_size = 92
cortexDepths = 5

s = statistics(name_cortex, name_otsu, path, path_directionality, patch_size)
plot_Statistics(name_data, path, s, patch_size, save_path)