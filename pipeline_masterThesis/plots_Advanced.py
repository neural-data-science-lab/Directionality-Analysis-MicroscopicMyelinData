'''
author: Gesine Mueller, 21-10-21
file to combine information from either Left/Right or fid differences between OrientationJ and Fiji_directionality
'''

# ToDo Left/Right polarLayer
# ToDo: create a plot substracting the differences between Fiji Directionality_() and my OrientationJ()


import cmocean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def layer(result):
    direction = result['5'][0][:, 0]
    d = pd.DataFrame(np.column_stack(
        (np.copy(direction), np.zeros((len(direction), len(layers))))))  # pd for corrected valid directionality frequencies
    nbr_d = pd.DataFrame(np.zeros((1, len(layers))))  # pd to sample for the nbr of patches per cortex depth
    for i in range(result.shape[0]):
        key_layer = np.digitize(result[4][i], layers, right=False)  # key position for layer I -> VI
        d[key_layer] += result[5][i][:, 1]
        nbr_d[key_layer - 1] += 1
    return d, nbr_d

def plot_directionalityPolar(patch_size, data_l, nbr_l, data_r, nbr_r, method, path_output):
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
        freq_l = data_l[i] / (nbr_l[i-1])[0]
        freq_r = data_r[i] / (nbr_r[i-1])[0]
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
    name = 'PolarLayer_' + str(patch_size) + method+'.png'
    plt.savefig(path_output+name, dpi = 200)
    plt.close()


def plot_nbrPatchesInCortex(patch_size, nbr_l,  nbr_r, method, path_output):
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
    plt.savefig(path_output + 'nbrPatches_'+str(patch_size)+method+'.png', dpi=200)
    plt.close()

################  MAIN
pixel = 0.5417  # um per pixel
side = ['Left', 'Right']
patch_size = int(round(92.25))
method = 'Fiji_Directionality_'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/test/Pipeline/'
layers = np.array([0, 58.5, 234.65, 302.25, 557.05])/pixel  #layer in um / pixel


result_left = pd.read_csv(path + 'Result_'+ side[0] + str(patch_size) + '_'+ method + '.csv', encoding = "ISO-8859-1", index_col=0)
result_right = pd.read_csv(path + 'Result_'+ side[1] + str(patch_size) + '_'+ method + '.csv', encoding = "ISO-8859-1", index_col=0)
d_l, nbr_l = layer(result_left)
d_r, nbr_r = layer(result_left)

# polarLayer L/R :ToDo Left/Right
plot_directionalityPolar(patch_size, d_l, nbr_l, d_r, nbr_r, method, path)
plot_nbrPatchesInCortex(patch_size, nbr_l, nbr_r, method, path)
