'''
author: Gesine Mueller, 21-10-21
file to combine information from either Left/Right or find differences between OrientationJ and Fiji_directionality
'''


import os
from skimage import io
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('path', type=str)
parser.add_argument('patch_size', type=int)
args = parser.parse_args()

def layer(result, distribution, layers):
    direction = pd.DataFrame(distribution['0'])
    d = pd.DataFrame(np.column_stack(
        (np.copy(direction), np.zeros((len(direction), len(layers))))))  # pd for corrected valid directionality frequencies
    nbr_d = pd.DataFrame(np.zeros((1, len(layers))))  # pd to sample for the nbr of patches per cortex depth
    for i in range(result.shape[0]):
        key_layer = np.digitize(result['4'][i], layers, right=False)  # key position for layer I -> VI
        d[key_layer] += distribution[str(i+1)]
        nbr_d[key_layer - 1] += 1
    return d, nbr_d

def domDir_layer_tonotopy(result, frangi_data, max_dist, patch_size):
    x_resolution = np.arange(0, max_dist - patch_size, patch_size)
    height = frangi_data.shape[1]
    s = pd.DataFrame(np.zeros((int(height / patch_size), len(x_resolution))))  # pd for corrected valid average orientation per layer and pos in tonotopic axis
    nbr_s = pd.DataFrame(np.zeros((int(height / patch_size), len(x_resolution))))  # pd to sample for the nbr per patches layer and position along tonotopic axis
    for i in range(result.shape[0]):
        key_x_resolution = np.digitize(result['4'][i], x_resolution, right=False) - 1  # key position for x_resolution
        key_tonotopy = result['1'][i]  # key for position in tonotopic axis
        s[key_x_resolution][key_tonotopy] += result['3'][i]
        nbr_s[key_x_resolution][key_tonotopy] += 1
    return s, nbr_s

def plot_directionalityPolar(patch_size, data_l, data_r, method, name, path_output):
    '''
    Plot 1D directionality per layer; comparison between left and right cortex
    patch_size:         size of patch on which the directionality distributions were computed
    data_l, data_r:     sum of all distributions for each layer left/right
    nbr_l, nbr_r:       nbr od patches per layer for normalization left/right
    method:             Fiji_directionality, OrientationJ
    path_output:        path to where to save the resulting image
    return:             polar plot of both sides per layer
    '''
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax1 = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122, polar=True)
    x_lim = 0
    labels = ['I', 'II/III', 'IV', 'V', 'VI']
    theta = np.deg2rad(np.array(data_l[0]))
    for i in (np.arange(1, data_l.shape[1],1)):
        freq_l = data_l[i] / np.sum(data_l[i])
        freq_r = data_r[i] / np.sum(data_r[i])
        ax1.plot(theta, np.array(freq_l))
        ax2.plot(theta, np.array(freq_r))
        max_lim = np.max(freq_l)
        if max_lim > x_lim:
            x_lim = max_lim
    title = 'Normalized directionality analysis'
    fig.suptitle(title, fontsize=14)
    ax1.set_thetamin(90)
    ax1.set_thetamax(-90)
    ax1.set_theta_zero_location("W")
    ax1.set_theta_direction(-1)
    ax2.set_thetamin(90)
    ax2.set_thetamax(-90)
    ax1.set_ylim(0, x_lim)
    ax1.invert_xaxis()
    ax2.set_ylim(0, x_lim)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax2.set_title('Right')
    ax1.set_title('Left')
    ax2.legend(['L1', 'L2/3', 'L4', 'L5', 'L6'])
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    save_name = 'PolarLayer_' + name + '_'+str(patch_size) +'_'+ method+'.png'
    #plt.savefig(path_output+save_name, dpi = 200)
    #plt.close()


def plot_nbrPatchesInCortex(patch_size, nbr_l,  nbr_r, method, name, path_output):
    '''
    PLot to display the nbr of patches summed over for the directionality analysis
    patch_size:         size of patch on which the directionality distributions were computed
    nbr_l, nbr_r:       nbr od patches per layer for normalization left/right
    method:             Fiji_directionality, OrientationJ
    path_output:        path to where to save the resulting image
    return:             bar plot of nbr of patches for both sides and layers
    '''
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=200)
    ax[0].bar(np.arange(0, nbr_l.shape[1], 1), nbr_l.values.ravel())
    ax[1].bar(np.arange(0, nbr_r.shape[1], 1), nbr_r.values.ravel())
    ax[0].invert_xaxis()
    ax[0].set_ylabel('# patches', fontsize=14)
    ax[0].set_xlabel('cortex depth', fontsize=14)
    ax[1].set_xlabel('cortex depth', fontsize=14)
    ax[0].set_xticks(np.arange(0,  nbr_l.shape[1], 1))
    ax[1].set_xticks(np.arange(0,  nbr_r.shape[1], 1))
    ax[0].set_ylim([0, round(max(max(nbr_l.max(axis=1)), max(nbr_r.max(axis=1))),-2)]) #-2 to come to next 100 for axis
    ax[0].set_xticklabels(np.array(['I', 'II/III', 'IV', 'V', 'VI']), fontsize=14)
    ax[1].set_xticklabels(np.array(['I', 'II/III', 'IV', 'V', 'VI']), fontsize=14)
    ax[1].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    ax[0].set_title('Left')
    ax[1].set_title('Right')
    fig.suptitle('Patches per cortex layer', fontsize=14)
    # save figure
    #plt.savefig(path_output + 'nbrPatches_'+name+'_'+str(patch_size)+'_'+method+'.png', dpi=200)
    #plt.close()


def plot_color2D_layerTonotopy(data, path_output, patch_size, side, annot, cmap = 'Oranges', pixel = 0.5417):
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
    sns.color_palette(cmap, as_cmap=True)
    sns.heatmap(data, cmap=cmap, square=True, xticklabels=False, yticklabels=False,
                vmin=0, vmax=90, center=0, cbar_kws={"shrink": .6}, annot=annot, annot_kws={"size": 5}) #
    ax1.set_xlim(ax1.get_xlim())
    ax2 = ax1.twiny()
    ax1.set_ylabel('Tonotopic axis', fontsize=20)
    ax1.set_xlabel('Layers', fontsize=20)
    layer_mid = np.array([0, 30, 60, 147.5, 235, 267.5, 300, 430, 560, 655, 750]) / pixel
    new_tick_locations = layer_mid / patch_size
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.05))
    ax2.set_xticks(new_tick_locations)
    Layers = ['', 'I', '', 'II/III', '', 'IV', '', 'V', '', 'VI', '']
    ax2.set_xticklabels(Layers, fontsize=16, horizontalalignment='center')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    if annot == True:
        plt.savefig(path_output +'DifferenceFijiOriJ_Layers_tonotopy_'+side+'_'+str(patch_size)+'_annon.png', dpi=180)
    else:
        plt.savefig(path_output + 'DifferenceFijiOriJ_Layers_tonotopy_' + side + '_' + str(patch_size) + '.png',
                    dpi=180)
    plt.close()

################  MAIN
pixel = 0.5417  # um per pixel
side = [args.name + '_l_ACx', args.name + '_r_ACx']
patch_size = int(round(args.patch_size))
method = ['Fiji_Directionality_', 'OrientationJ_']
layers = np.array([0, 58.5, 234.65, 302.25, 557.05])/pixel  #layer in um / pixel
max_dist = 752.05 / pixel


# polarLayer L/R
for i in range(len(method)):
    result_left = pd.read_csv(args.path + side[0] + '/' + 'Result_'+ side[0] +'_'+ str(patch_size) + '_'+ method[i] + '.csv', encoding = "ISO-8859-1")
    result_right = pd.read_csv(args.path + side[1] + '/' + 'Result_'+ side[1] +'_'+ str(patch_size) + '_'+ method[i] + '.csv', encoding = "ISO-8859-1")
    distribution_left = pd.read_csv(args.path + side[0] + '/' + 'Distribution_'+ side[0] +'_'+ str(patch_size) + '_'+ method[i] + '.csv', encoding = "ISO-8859-1")
    distribution_right = pd.read_csv(args.path + side[1] + '/' + 'Distribution_'+ side[1] +'_'+ str(patch_size) + '_'+ method[i] + '.csv', encoding = "ISO-8859-1")
    d_l, nbr_l = layer(result_left, distribution_left, layers)
    d_r, nbr_r = layer(result_right, distribution_right, layers)
    plot_directionalityPolar(patch_size, d_l, d_r, method[i], args.name, args.path)
    plot_nbrPatchesInCortex(patch_size, nbr_l, nbr_r, method[i], args.name, args.path)


# absolute difference between Fiji_directionality and OrientationJ orientations (dominant direction)
for i in range(len(side)):
    data_frangi = side[i] + '/' + side[i] + '_C03_bg.tif'
    frangi_data = io.imread(os.path.join(args.path, data_frangi))
    result_left = pd.read_csv(args.path + side[i] + '/' + 'Result_'+ side[i] +'_'+ str(patch_size) + '_'+ method[0] + '.csv', encoding = "ISO-8859-1")
    result_right = pd.read_csv(args.path + side[i] + '/' + 'Result_'+ side[i] +'_'+ str(patch_size) + '_'+ method[1] + '.csv', encoding = "ISO-8859-1")
    s_left, nbr_left = domDir_layer_tonotopy(result_left, frangi_data, max_dist, patch_size)
    s_right, nbr_right = domDir_layer_tonotopy(result_right, frangi_data, max_dist, patch_size)
    data = np.abs(np.abs(s_left/nbr_left)-np.abs(s_right/nbr_right))
    plot_color2D_layerTonotopy(data, args.path, patch_size, side[i], True, cmap = 'RdGy')
    plot_color2D_layerTonotopy(data, args.path, patch_size, side[i], False, cmap = 'RdGy')
