#### Plots (later for publications) using seaborn
import numpy as np
import math
import os
import skimage.io as io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_Statistics(name_data, path, statistics, patch_size, save_path, slice=0):
    '''
    Plot to have an overlay of frangi dataset and modes from statistics
    require mode from all patches of one z-slice (depth): see Statistics_only.py
    '''
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

def plot_directionalityCorreted(patch_size, data_l, nbr_l, data_r, nbr_r, save_path, normalize = False):
    '''
    Plot 1D directionality per layer; comparison between left and right cortex
    choice between normalization or not with the nbr of patches per layer
    '''
    labels = nbr_l.keys()
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=200) #, sharex=True, sharey=True
    s_l = np.sum(nbr_l.values)
    s_r = np.sum(nbr_r.values)
    x_lim = 0
    for i, l in enumerate(labels):
        if normalize:
            title = 'Normalized directionality analysis'
            freq_l = data_l[l] / (nbr_l[l])[0]
            freq_r = data_r[l] / (nbr_r[l])[0]
            name = 'Directionality_norm_'+str(patch_size)+'.png'
            max_lim = np.max(freq_l)
            if max_lim > x_lim:
                x_lim = max_lim
            ax1.set_xlim([x_lim+0.001, 0])
            ax1.invert_xaxis()
            ax2.set_xlim([0, x_lim+0.001])
            #x_lim = np.ceil(np.max(np.array([np.max(data_l.values)*nbr_l[data_l.columns[data_l.isin([np.max(data_l.values)]).any()][0]][0]/s_l,
                                             #np.max(data_r.values)*nbr_r[data_r.columns[data_r.isin([np.max(data_r.values)]).any()][0]][0]/s_r])))
        else:
            freq_l = data_l[l]
            freq_r = data_r[l]
            title = 'Directionality analysis'
            name = 'Directionality'+str(patch_size)+'.png'
            #x_lim = round(np.max(np.max(data_l)), -3)
            x_lim = np.ceil(np.max(np.array([np.max(data_l.values), np.max(data_r.values)])))
            ax1.set_xlim([x_lim, 0])
            ax1.invert_xaxis()
            ax2.set_xlim([0, x_lim])
        ax1.plot(np.array(freq_l), np.array(data_l['Direction']), label='layer ' + l)
        ax2.plot(np.array(freq_r), np.array(data_r['Direction']), label='layer ' + l)
    ax1.set_ylabel('Directions in angluar degrees', fontsize=14)
    ax1.set_xlabel('Frequency', fontsize=14)
    ax1.set_title('Left')
    ax1.invert_xaxis()
    fig.suptitle(title, fontsize=14)
    ax1.legend(fontsize=14)
    ax2.set_yticks([])
    ax2.set_xlabel('Frequency', fontsize=14)
    ax2.set_title('Right')
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tight_layout()
    plt.show()
    # save figure
    plt.savefig(save_path+name, dpi = 200)

def plot_directionalityPolar(patch_size, data_l, nbr_l, data_r, nbr_r, save_path):
    '''
        Plot 1D directionality per layer; comparison between left and right cortex
        polar version of plot_directionalityCorreted
            choice between normalization or not with the nbr of patches per layer

        '''
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax1 = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122, polar=True)
    x_lim = 0
    labels = nbr_l.keys()
    theta = np.deg2rad(np.array(data_l['Direction']))
    for i, l in enumerate(labels):
        freq_l = data_l[l] / (nbr_l[l])[0]
        freq_r = data_r[l] / (nbr_r[l])[0]
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
    name = 'Directionality_norm_polar' + str(patch_size) + '.png'
    plt.savefig(save_path+name, dpi = 200)


def plot_nbrPatchesInCortex(patch_size, nbr_l, nbr_r, save_path):
    '''
    PLot to display the nbr of patches summed over for the directionality analysis
    '''
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=200)
    ax[0].bar(np.arange(0, len(nbr_l.keys()), 1), nbr_l.values.ravel())
    ax[1].bar(np.arange(0, len(nbr_r.keys()), 1), nbr_r.values.ravel())
    ax[0].invert_xaxis()
    ax[0].set_ylabel('# patches', fontsize=14)
    ax[0].set_xlabel('cortex depth', fontsize=14)
    ax[1].set_xlabel('cortex depth', fontsize=14)
    ax[0].set_xticks(np.arange(0, len(nbr_l.keys()), 1))
    ax[1].set_xticks(np.arange(0, len(nbr_r.keys()), 1))
    ax[0].set_ylim([0, round(max(max(nbr_l.max(axis=1)), max(nbr_r.max(axis=1))),-3)])
    ax[0].set_xticklabels(np.array(['I', 'II/III', 'IV', 'V', 'VI']), fontsize=14)
    ax[1].set_xticklabels(np.array(['I', 'II/III', 'IV', 'V', 'VI']), fontsize=14)
    ax[1].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    ax[0].set_title('Left')
    ax[1].set_title('Right')
    fig.suptitle('Patches per cortex layer', fontsize=14)
    # save figure
    plt.savefig(save_path + 'nbrPatches_'+str(patch_size)+'.png', dpi=200)


# main
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/'
patch_size = 20
folder_left = 'Left_frangi_'+str(patch_size)+'/'
import_left = path + folder_left
folder_right = 'Right_frangi_'+str(patch_size)+'/'
import_right = path + folder_right
#corrected = pickle.load(open(import_path + "corrected.pkl", "rb"))
#nbr = eval(open(import_path + "nbr.json", "r").read())
layers = ['I', 'II/III', 'IV', 'V', 'VI']
nbr_l = pd.read_csv(import_left + 'n.csv', names = layers)
nbr_l = nbr_l.drop(nbr_l.index[[0]])
nbr_r = pd.read_csv(import_right + 'n.csv', names = layers)
nbr_r = nbr_r.drop(nbr_r.index[[0]])

layers.insert(0, 'Direction')
data_l = pd.read_csv(import_left + 'd.csv', names = layers)
data_l = data_l.drop(data_l.index[[0]])
data_r = pd.read_csv(import_right + 'd.csv', names = layers)
data_r = data_r.drop(data_r.index[[0]])

plot_directionalityCorreted(patch_size, data_l, nbr_l, data_r, nbr_r, path, normalize = False)
plot_directionalityCorreted(patch_size, data_l, nbr_l, data_r, nbr_r, path, normalize = True)
plot_nbrPatchesInCortex(patch_size, nbr_l, nbr_r, path)
plot_directionalityPolar(patch_size, data_l, nbr_l, data_r, nbr_r, path)


############################################### Statistics Levy #######################################################
def plot_color2D_layerTonotopy(stats, nbr, save_path, patch_size, cmap = 'PuOr', pixel = 0.542):
    '''
    PLot ala Levy2019 3b/c with the axes: layers and tonotopic axis
    Mode of orientations of patches are averaged over the z-depth and normalized by the nbr of patches per layer & tonotopic axis
    '''
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 15))
    fig.subplots_adjust(bottom=0.2)
    x_axis_labels = ['I', 'II/III', 'IV', 'V', 'VI']  # labels for x-axis
    sns.color_palette("mako", as_cmap=True)
    sns.heatmap(stats / nbr, cmap=cmap, square=True, xticklabels=False, yticklabels=False,
                vmin=-90, vmax=90, center=0, cbar_kws={"shrink": .6},
                annot=True, annot_kws={"size": 5})  # np.min(np.min(stats / nbr)), np.max(np.max(stats / nbr))
    ax1.set_xlim(ax1.get_xlim())
    ax2 = ax1.twiny()
    ax1.set_ylabel('Tonotopic axis', fontsize=20)
    ax1.set_xlabel('Layers', fontsize=20)
    pixel = 0.542
    layers = np.array([0, 60, 235, 300, 560]) / pixel
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
    # ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    plt.savefig(save_path + cmap+ 'Layers_tonotopy.png', dpi=180)
    plt.close()

path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/test/' #'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/'
data = path + 'dir_92/' #'Right_frangi_40/'
#right = path + 'Right_frangi_40/'
save_path = data

patch_size = 92
Layers = ['I', 'II/III', 'IV', 'V', 'VI']
stats = pd.read_csv(data + 's.csv')
stats = stats.drop('Unnamed: 0', axis = 1)
nbr = pd.read_csv(data  + 'nbr.csv')
nbr = nbr.drop('Unnamed: 0', axis = 1)
plot_color2D_layerTonotopy(stats, nbr, save_path, patch_size)

