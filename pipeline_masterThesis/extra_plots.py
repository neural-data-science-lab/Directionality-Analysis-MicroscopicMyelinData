import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import io, feature
from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import argparse


def plot_color2D_layerTonotopy(stats, nbr, path_output, patch_size, method, name, annontation, cmap, side, pixel = 0.5417):
    '''
    Plot see Levy2019 3b/c with the axes: layers and tonotopic axis
    Mode of orientations of patches are averaged over the z-depth and normalized by the nbr of patches per layer & tonotopic axis
    :param stats:       dominant direction per x-resolution and tonotopic axis resolution, averaged over z-dimension
    :param nbr:         nbr of patches per resolution
    :param path_output: path to where to save the resulting image
    :param patch_size:  size of patch on which the directionality distributions were computed
    :param method:      Fiji_directionality, OrientationJ
    :param cmap:        color_map to choose from
    :param pixel:       1 pixel = 0.5417 um
    :return: plot per tonotopic axis and x-resolution along layer depth
    '''
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 15))
    fig.subplots_adjust(bottom=0.2)
    x_axis_labels = ['I', 'II/III', 'IV', 'V', 'VI']  # labels for x-axis
    sns.color_palette("twilight_r", as_cmap=True)
    sns.heatmap(stats / nbr, ax=ax1, cmap=cmap, square=True, xticklabels=False, yticklabels=False,
                vmin=-90, vmax=90, center=0, cbar_kws={"shrink": .8}, annot=annontation, annot_kws={"size": 10})  #
    ax1.xaxis.tick_top()
    ax1.set_ylabel('Tonotopic axis', fontsize=24)
    ax1.xaxis.set_label_position('top')
    ax1.set_xlabel('Layer', fontsize=24)
    layer_mid = np.array([0, 29.25, 58.5, 146.575, 234.65, 267.95, 302.25, 429.65, 557.05, 654.55, 752.05]) / pixel
    new_tick_locations = layer_mid / patch_size
    ax1.set_xticks(new_tick_locations)
    Layers = ['', 'I', '', 'II/III', '', 'IV', '', 'V', '', 'VI', '']
    ax1.set_xticklabels(Layers, fontsize=22, horizontalalignment='center')
    if side == 'r':
        plt.gca().invert_xaxis()
    cbar = ax1.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)
    if annontation == True:
        plt.savefig(path_output +'Layers_tonotopy_'+name+str(patch_size)+method+'annot.png', dpi=180)
    else:
        plt.savefig(path_output + 'Layers_tonotopy_' + name + str(patch_size) + method + '.png', dpi=180)
    #plt.close()


path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/stats_C3/'
result = pd.read_csv(path + 'Result_PR014_r_VCx_92_Fiji_Directionality_.csv')
method = 'Fiji_Directionality_'
name = 'PR014_r_VCx'

result = pd.DataFrame(result) # [(k,j,i), dominant direction, cortex depth, correction factor]

max_dist = 752.05 / 0.5417
patch_size = 92
x_resolution = np.arange(0, max_dist-patch_size, patch_size)
height = 2200
s = pd.DataFrame(np.zeros((int(height/patch_size), len(x_resolution))))  # pd for corrected valid average orientation per layer and pos in tonotopic axis
nbr_s = pd.DataFrame(np.zeros((int(height/patch_size), len(x_resolution))))  # pd to sample for the nbr per patches layer and position along tonotopic axis
for i in range(result.shape[0]):
    key_x_resolution = np.digitize(result['4'][i], x_resolution, right=False)-1  #key position for x_resolution
    key_tonotopy = result['1'][i]  # key for position in tonotopic axis
    s[key_x_resolution][key_tonotopy] += result['3'][i]
    nbr_s[key_x_resolution][key_tonotopy] += 1
plot_color2D_layerTonotopy(s, nbr_s, path, patch_size, method, name, annontation = False, cmap = 'twilight_r', side = 'r', pixel = 0.5417)
#plot_color2D_layerTonotopy(s, nbr_s, path, patch_size, method, name, annontation = True, cmap = 'twilight_r', pixel = 0.5417)






