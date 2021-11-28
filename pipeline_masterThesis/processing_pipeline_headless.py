'''
author: Gesine Müller 12-10-21
Creating a python pipeline spanning every step from data-preprocessing to the directionality pipeline, everything in batches:
    - Input: ACx.hdf5 raw data from detail dataset, myelin channel (after registration, after stitching):
      uint16, 0.5417x0.5417x6um --> Input path to that data, all rest: outputpath
    - pre-process dataset: Gaussian blur, background reduction, vesselness filter -> 3D stack
    - creating masks: threshold mask (on pre-processed dataset), cortex mask (autofluorescence channel) -> 3D stacks
    - Directionality analysis: performing the orientationJ method on pre-processed dataset
    - Statistics: visualization of one slice with major direction
    - Visualization: Levy, ...
'''

import os
import math
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import io, feature
from skimage.io import imsave
from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import white_tophat, disk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('path', type=str)
parser.add_argument('patch_size', type=int)
parser.add_argument('batch_size', type=int)
parser.add_argument('pre_processing_total', type=bool)
parser.add_argument('vesselness', type=bool)
parser.add_argument('orientationJ', type=bool)
parser.add_argument('Diretionality', type=bool)
parser.add_argument('plots', type=bool)
args = parser.parse_args()


def normalize(image):
    min_val=np.min(image)
    max_val=np.max(image)
    return (image-min_val)/(max_val-min_val)


############################### preprocessing using either fiji or py libraries #######################################
def PreProcessing_Masks(myelin, autofluorescence):
    '''
    function that takes ACx raw data from detail dataset and creates the pre-processed dataset + otsu and cortex masks
    for directionality analysis
        1. Gaussian blur, background reduction, vesselness filter on myelin channel
        2. Creating masks: threshold mask (on pre-processed dataset, myelin), cortex mask (autofluorescence channel)

    :param myelin: name of myelin channel dataset
    :param autofluorescence: name of autofluoresecnce channel dataset
    :return: frangi_data, cortex mask and otsu mask
    '''
    g_filter = gaussian_filter(myelin, sigma=2.0, mode='nearest')
    g_autofl = gaussian_filter(autofluorescence, sigma=25, mode='nearest')

    bg_filter = []  # rolling ball discussion: https://forum.image.sc/t/rolling-ball-background-subtraction-challenge/31837/5
    frangi_data = []
    otsu_mask = []
    cortex_mask = []

    for z in range(myelin.shape[0]):
        bg_filter.append(white_tophat(g_filter[z], disk(radius=35)))
        frangi_data.append(frangi(bg_filter[z], black_ridges=False))
        otsu_mask.append(bg_filter[z] > threshold_otsu(bg_filter[z]))
        cortex_mask.append(g_autofl[z] > threshold_otsu(g_autofl[z]))

    frangi_data = np.stack((normalize(frangi_data) * 65535).astype('uint16'))
    otsu_mask = np.stack(binary_fill_holes(otsu_mask)).astype('uint16')  #0 or 1
    cortex_mask = np.stack(binary_fill_holes(cortex_mask)).astype('uint16') #file hole + rescaling

    return frangi_data, otsu_mask, cortex_mask


def Vesselness(bg):
    '''
    function that computes frangi vesselness filter on bg_data (gaussian filter + bg reduction via Fiji)
    :param bg: bg_dataset
    :return: frangi_data
    '''
    frangi_data = []
    for z in range(bg.shape[0]):
        frangi_data.append(frangi(bg[z], black_ridges=False))

    frangi_data = np.stack((normalize(frangi_data) * 65535).astype('uint16'))

    return frangi_data



################################### Orientation distribution (Fiji or Py here) #########################################
def OrientationJ(patch, sigma=2):
    '''
    function to compute the orientation distribution and dominant direction of an image (3D) via the structure tensor
    (comparable to the OrientationJ implementation: http://bigwww.epfl.ch/demo/orientation/, https://forum.image.sc/t/orientationj-or-similar-for-python/51767/3)
    :param patch: input image 3D
    :param sigma: std for the Gaussian used in the structure tensor
    :return: histogram of the orientations found, excluding 0, dominant direction as the mean of the distribution
    '''
    direction = np.arange(-90,90,2.022471909999993) #comparable to the fiji directionality Plugin with nbins=90,start=-90
    dom_ori = []
    hist_ori = []
    hist_ori.append(direction)
    dim = patch.shape[0]
    for i in range(dim):
        axx, axy, ayy = feature.structure_tensor(patch[i].astype(np.float32), sigma=sigma, mode="reflect", order="xy")
        o = np.rad2deg(np.arctan2(2 * axy, (ayy - axx)) / 2)
        o = o[o != 0]   #delete zeros from array in order to have 0 as the mode direction always
        if o.size == 0:
            h = np.empty(len(direction))
            h.fill(np.nan)
            hist_ori.append(h)
            dom_ori.append(np.nan)
        else:
            d = np.digitize(o, direction, right=True)
            h = np.histogram(d, bins=90, range=(0, 89))[0]
            hist_ori.append(h / np.sum(h))
            dom_ori.append(np.rad2deg(np.mean(o) % (2 * math.pi)) - 180)
    dom_ori = np.stack(dom_ori, axis=0)
    ori = np.stack(hist_ori, axis=1)

    return dom_ori, ori


################################################ Directionality analsis #############################################
def directionality_analysis(otsu_mask, cortex_mask, path, name_orientation, batch_size, patch_size, colnames,
                                  header, binary, pixel=0.5417):
    '''
    function to:
    1. extract all valid patches in the sense that based on a binary mask only those orientation patches are valid in
    which the respective mask patch is not 0;
    2. rotate the orientations from directionality calculation in order to respect cortex curvature
    (Gradient filter over distance transform)
    3. save the corrected distributions
    4. create statistics and calculate the mode in each patch
    5. save everything to list or in case of the distribution in a pandas frame

    otsu_mask:              threshold mask -> defines valid patches
    cortex_mask:            cortex mask -> cortex curvature
    path:                   path to directionality files
    name_orientation:       name of csv. files including the method (Fiji directionality, OrientationJ.py)
    batch_size:             size of batch in which the data is split for memory issues
    patch_size:             size of square patch on which the orientation was computed
    colnames:               have the same dataframe from fiji directionality and OrientationJ.py
    header:                 if True: fiji directionality, if False: OrientationJ.py
    pixel:                  1 pixel = 0.5417 um

    returns list result: [position in dataset (k,j,i), dominant direction as the mode of the corrected distribution,
    cortexDepth, corrected distribution, correction factor for cortex normal]
    '''

    width = cortex_mask.shape[2]
    height = cortex_mask.shape[1]
    depth = cortex_mask.shape[0]
    max_dist = 752.05 / pixel

    # initialize the sum over the directionality
    file = name_orientation + str(0) + '_' + str(0) + '.csv'
    path_patch0 = os.path.join(path, file)
    patch0 = pd.read_csv(path_patch0, encoding = "ISO-8859-1")
    if not header:
        patch0.columns = colnames
    else:
        patch0.rename(columns={'Direction (°)': 'Direction'}, inplace=True)
    direction = patch0['Direction']

    result = []
    distribution_corrected = []
    distribution_corrected.append(direction)

    for batch in range(0, int(depth / batch_size)):
        orientations = []
        dists3D = distance_transform_edt(cortex_mask[batch * batch_size:batch * batch_size + batch_size, :, :],
                                                 sampling=[8, 1, 1], return_distances=True)

        for z in range(batch_size):
            sx = ndimage.sobel(dists3D[z], axis=0, mode='nearest')
            sy = ndimage.sobel(dists3D[z], axis=1, mode='nearest')
            sobel = np.arctan2(sy, sx) * 180 / np.pi
            sobel_smooth = gaussian_filter(sobel, sigma=2)
            orientations.append(sobel_smooth)  # angles difference between cortex normal and 0° (East)

        for i in range(int(width / patch_size)):
            for j in range(int(height / patch_size)):
                filename = name_orientation + str(i) + '_' + str(j) + '.csv'
                path_patch = os.path.join(path, filename)
                patch = pd.read_csv(path_patch, encoding = "ISO-8859-1")
                if not header:
                    patch.columns = colnames
                else:
                    patch.rename(columns={'Direction (°)': 'Direction'}, inplace=True) #�

                x = np.arange(batch * batch_size, batch * batch_size + batch_size, 1) #z-slices according to batch to be considered
                for k, v in enumerate(x):
                    patch_otsu = otsu_mask[v, j * patch_size:j * patch_size + patch_size,
                                 i * patch_size:i * patch_size + patch_size]
                    cortexDepth = dists3D[k][int(j * patch_size + patch_size / 2), int(i * patch_size + patch_size / 2)]
                    if binary in patch_otsu and cortexDepth <= max_dist and np.isnan(np.min(patch['Slice_' + str(v + 1)])) == False: #255 for mask fiji  else 1
                        angle_cortex = orientations[k][int(j * patch_size + patch_size / 2),
                                                       int(i * patch_size + patch_size / 2)]
                        # get angle difference and rotate all orientations in patch
                        correction = 90 - angle_cortex
                        direction_corrected = patch['Direction'] - correction
                        # shift angles < -90 and > 90 degrees back into -90 to 90 range
                        patch_shifted = pd.concat([direction_corrected, patch['Slice_' + str(v + 1)]], axis=1)
                        patch_shifted['Direction'].loc[patch_shifted['Direction'] < -90] += 180
                        patch_shifted['Direction'].loc[patch_shifted['Direction'] > 90] -= 180
                        # relocate directionality values after shifting to original -90/90 interval ->take nearest value in 'Direction' and save in summary
                        distribution = np.stack((np.copy(direction), np.zeros(len(direction))), axis=1) #corrected distriution for i,j,v
                        for row in range(len(patch_shifted)):
                            idx = (np.abs(distribution[:,0] - patch_shifted['Direction'][row])).argmin()
                            distribution[idx,1] = patch_shifted['Slice_' + str(v + 1)][row]
                        domDir = distribution[distribution[:, 1].argmax(), 0]  #get mode of directions in patch
                        result.append(np.array([v, j, i, domDir, cortexDepth, correction]))
                        distribution_corrected.append(distribution[:, 1])
    return result, distribution_corrected



############################################# Directionality vizualizations ##########################################
def plot_color2D_layerTonotopy(stats, nbr, path_output, patch_size, method, name, cmap = 'PuOr', pixel = 0.5417):
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
    sns.color_palette("mako", as_cmap=True)
    sns.heatmap(stats / nbr, cmap=cmap, square=True, xticklabels=False, yticklabels=False,
                vmin=-90, vmax=90, center=0, cbar_kws={"shrink": .6}, annot=True, annot_kws={"size": 5}) #
    ax1.set_xlim(ax1.get_xlim())
    ax2 = ax1.twiny()
    ax1.set_ylabel('Tonotopic axis', fontsize=20)
    ax1.set_xlabel('Layers', fontsize=20)
    layer_mid = np.array([0, 29.25, 58.5, 146.575, 234.65, 267.95, 302.25, 429.65, 557.05, 654.55, 752.05]) / pixel
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

    plt.savefig(path_output +'Layers_tonotopy_'+name+str(patch_size)+method+'.png', dpi=180)
    plt.close()


def plot_domOrientation(frangi_data, path_output, domDir, method, patch_size, slice, name):  # cividis, PuOr,
    '''
    Plot to have a visual validation about whether the directionality found follows the myelin structures
    :param frangi_data:     pre-processed myelin data
    :param path_output:     path to where to save the resulting image
    :param domDir:          dominant directions
    :param method:          Fiji_directionality, OrientationJ
    :param patch_size:      size of patch on which the directionality distributions were computed
    :param slice:           z-depth
    :return: 2D image of an overlay of the frangi_data and the vectorfield of the dominant directions
    '''
    data = frangi_data[slice]
    X = domDir[2] * patch_size + patch_size / 2
    Y = domDir[1] * patch_size + patch_size / 2
    angles = domDir[3] + domDir[5]  # mode orientation + correction
    angles.loc[angles < -90] += 180  # rescale to -90° -> 90°
    angles.loc[angles > 90] -= 180
    U = np.cos(angles * np.pi / 180)
    V = np.sin(angles * np.pi / 180)
    colors = np.array(np.arctan2(V, U))
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.PuOr
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(data, cmap="Greys")
    ax.quiver(X, Y, U, V, color=colormap(norm(colors)), units='xy', linewidths=12)
    norm.autoscale(np.array(angles))
    sm = plt.cm.ScalarMappable(cmap='PuOr', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, fraction=0.035)
    # ax.quiver(X, Y, U, V, color='red', units='xy')
    # save figure
    plt.savefig(path_output + 'domOrientation_' + name +str(patch_size)+ method + 'Slice'+ str(slice) + '.png', dpi=200)
    plt.close()




######################################################## MAIN ########################################################
######################################## define batch-size, patch-size load data #####################################
#patch_size 18.45 ~ 10um, 27.67 ~ 15 um, 36.90 ~ 20um, 46.13 ~ 25um, 73.80 ~ 40 um, 92.25 ~ 50um, 138.38 ~ 75um
data = args.name + ".h5"
data_frangi = args.name + '_C03_frangi.tif'
data_otsu = args.name + '_C03_otsu.tif'
data_cortex = args.name + '_C00_cortex.tif'
data_gauss = args.name + '_C03_gauss.tif'

pixel = 0.5417  # um per pixel in x-y
colnames = pd.read_csv(os.path.join(args.path, 'colnamesFiji_Directionality.csv'))
colnames = colnames.values.astype('object')
colnames = colnames.flatten()
layers = np.array([0, 58.5, 234.65, 302.25, 557.05])/pixel  #layer in um / pixel



####################################################### main #########################################################
if args.pre_processing_total:
    myelin = h5py.File(os.path.join(args.path, data), 'r')[u"/t00000/s03/0/cells"]
    autofluorescence = h5py.File(os.path.join(args.path, data), 'r')[u"/t00000/s00/0/cells"]
    frangi_data = []
    otsu_mask = []
    cortex_mask = []
    for batch in range(int(myelin.shape[0]/args.batch_size)):
        f, o, c = PreProcessing_Masks(myelin[batch * args.batch_size:batch * args.batch_size + args.batch_size],
                                      autofluorescence[batch * args.batch_size:batch * args.batch_size + args.batch_size])
        frangi_data.append(f)
        otsu_mask.append(o)
        cortex_mask.append(c)
    frangi_data=np.vstack(frangi_data)
    otsu_mask=np.vstack(otsu_mask)
    cortex_mask=np.vstack(cortex_mask)
    # save finished dataset to output path: next step -> directionality (fiji or OrientationJ)
    imsave(args.path+args.side + "_" +"C03_frangi.tif", frangi_data.astype('uint16'))
    imsave(args.path+args.side + "_" +"C03_otsu.tif", otsu_mask.astype('uint16'))
    imsave(args.path+args.side + "_" +"C00_cortex.tif", (cortex_mask).astype('uint16'))
    binary = 1

elif args.vesselness:
    bg_data = io.imread(os.path.join(args.path, data_gauss))
    frangi_data = []
    for batch in range(int(bg_data.shape[0] / args.batch_size)):
        f = Vesselness(bg_data[batch * args.batch_size:batch * args.batch_size + args.batch_size])
        frangi_data.append(f)
    frangi_data = np.vstack(frangi_data)
    imsave(args.path + args.name + "_" + "C03_frangi.tif", frangi_data.astype('uint16'))
    otsu_mask = io.imread(os.path.join(args.path, data_otsu))
    cortex_mask = io.imread(os.path.join(args.path, data_cortex))
    binary = 255

else:
    frangi_data = io.imread(os.path.join(args.path, data_frangi))
    otsu_mask = io.imread(os.path.join(args.path, data_otsu))
    cortex_mask = io.imread(os.path.join(args.path, data_cortex))
    binary = 255

if args.orientationJ:
    header = False
    method = 'OrientationJ_'
    x_bound = int(frangi_data.shape[2]/args.patch_size)
    y_bound = int(frangi_data.shape[1]/args.patch_size)
    for i in range(x_bound):
        d = []
        for j in range(y_bound):
            patch = frangi_data[:, j*args.patch_size:j*args.patch_size+args.patch_size, i*args.patch_size:i*args.patch_size+args.patch_size]
            dom_dir, ori = OrientationJ(patch, sigma=2)
            ori = pd.DataFrame(ori)
            ori.to_csv(args.path+args.side+str(args.patch_size)+'_OrientationJ_'+str(i)+'_'+str(j)+'.csv', index=False)
            d.append(dom_dir)
        d = pd.DataFrame(d)
        #d.to_csv(path_output+side+str(patch_size)+'_domOrientationJ_' + str(i) + '.csv', index=False)

    name_orientation = args.side + str(args.patch_size) + '_' + method
    # [position in dataset (k,j,i), dominant direction, cortex depth, distribution, correction factor for cortex normal]
    result, distribution_corrected = directionality_analysis(otsu_mask, cortex_mask, args.path, name_orientation, args.batch_size, args.patch_size,
                                     colnames, header, binary, pixel=0.5417)
    pd.DataFrame(result).to_csv(args.path + 'Result'+ args.side + str(args.patch_size) + '_'+ method + '.csv', index=False)  # test = pd.read_csv(path_output+ method + 'result.csv', encoding = "ISO-8859-1")
    pd.DataFrame(np.stack(distribution_corrected, axis = 1)).to_csv(args.path + 'Distribution_' + args.side + str(args.patch_size) + '_' + method + '.csv', index=False)


if args.Directionality:
    header = True
    method = 'Fiji_Directionality_'
    name_orientation = args.side + str(args.patch_size) + '_' + method
    # [position in dataset (k,j,i), dominant direction, cortex depth, distribution, correction factor for cortex normal]
    result, distribution_corrected = directionality_analysis(otsu_mask, cortex_mask, args.path, name_orientation, args.batch_size, args.patch_size,
                                     colnames, header, binary, pixel=0.5417)
    pd.DataFrame(result).to_csv(args.path + 'Result_'+ args.side + str(args.patch_size) + '_'+ method + '.csv', index=False)
    pd.DataFrame(np.stack(distribution_corrected, axis = 1)).to_csv(args.path + 'Distribution_' + args.side + str(args.patch_size) + '_' + method + '.csv', index=False)




###################################################### main: Plots ####################################################
if args.plots:
    # domOrientation
    slice = np.arange(0,frangi_data.shape[0],10)
    for i in range(len(slice)):
        domDir = filter(lambda c: c[0] == slice[i], result)
        domDir = pd.DataFrame(list(domDir))
        plot_domOrientation(frangi_data, args.path, domDir, method, args.patch_size, slice[i], args.name)

    result = pd.DataFrame(result) # [(k,j,i), dominant direction, cortex depth, correction factor]

    # layerTonotopy
    max_dist = 752.05 / pixel
    x_resolution = np.arange(0, max_dist-args.patch_size, args.patch_size)
    height = frangi_data.shape[1]
    s = pd.DataFrame(np.zeros((int(height/args.patch_size), len(x_resolution))))  # pd for corrected valid average orientation per layer and pos in tonotopic axis
    nbr_s = pd.DataFrame(np.zeros((int(height/args.patch_size), len(x_resolution))))  # pd to sample for the nbr per patches layer and position along tonotopic axis
    for i in range(result.shape[0]):
        key_x_resolution = np.digitize(result[4][i], x_resolution, right=False)-1  #key position for x_resolution
        key_tonotopy = result[1][i]  # key for position in tonotopic axis
        s[key_x_resolution][key_tonotopy] += result[3][i]
        nbr_s[key_x_resolution][key_tonotopy] += 1
    plot_color2D_layerTonotopy(s, nbr_s, args.path, args.patch_size, method, args.name, cmap = 'PuOr', pixel = 0.5417)
