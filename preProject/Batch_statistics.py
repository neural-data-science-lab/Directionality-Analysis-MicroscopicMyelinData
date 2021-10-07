'''
author: Gesine Müller 26-06-21
Reproduce Fig 3 b/c from Levy2019 with the mode as the value (scale bar): x: tonotopic axis, y: Layers,
patch: average along the z-axis; correct with nbr. of valid patches
'''

import os
import math
import numpy as np
import pandas as pd
import skimage.io as io
import timeit
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import argparse

'''parser = argparse.ArgumentParser()
parser.add_argument('side', type=str)
parser.add_argument('patch_size', type=int)
args = parser.parse_args()'''

def directionality_layer_tonotopy(name_otsu, name_cortex, path, path_directionality, patch_size, colnames, header,
                                  pixel=0.542):
    '''
    1. extract all valid patches in the sense that based on a binary mask only those orientation patches are valid in
    which the respective mask patch is not 0;
    2. rotate the orientations from directionality calculation in order to respect cortex curvature
    (Gradient filter over distance transform)
    3. create statistics and calculate the mode in each patch
    4. sum over all patches with a certain cortex depth and position along the tonotopic axis

    name_otsu:              file name of the threshold mask, test_C03_smooth3D_bg95_otsu.tif -> valid patches
    name_cortex:            file name of the cortex mask, test_C00_binMask_cortex.tif -> cortex curvature
    path:                   path to  files
    patch_directionality:   path to where the directionality calculation files lie
    patch_size:             size of square patch on which the orientation was computed
    pixel:                  1 pixel = 0.542 um

    returns pd s: that contains the valid, corrected sums of mode orientations for specified layer and position in tonotopic axis
    pd n: nbr of valid patches for the normalization step later on
    '''
    path_cortex = os.path.join(path, name_cortex)
    mask_cortex = io.imread(path_cortex)
    path_otsu = os.path.join(path, name_otsu)
    mask_otsu = io.imread(path_otsu)
    width = mask_cortex.shape[2]
    height = mask_cortex.shape[1]
    depth = mask_cortex.shape[0]

    max_dist = 750 / pixel
    #layers = np.array([0, 60, 235, 300, 560]) / pixel  # starting value of the different layers in um -> pixel value
    layers = np.arange(0, max_dist-patch_size, patch_size)   # for plot for patch size
    layers_mid = layers + patch_size/2

    # initialize the sum over the directionality
    file = path_directionality + str(0) + '_' + str(0) + '.csv'
    path_patch0 = os.path.join(path, file)
    patch0 = pd.read_csv(path_patch0, encoding = "ISO-8859-1")
    if not header:
        patch0.columns = colnames
    patch0.rename(columns={'Direction (°)': 'Direction'}, inplace=True)
    direction = patch0['Direction']

    # create dataframe contain the nbr per layer and position along tonotopic axis and sum over statistics = average direction per patch
    nbr = pd.DataFrame(np.zeros((int(height/patch_size), len(layers))))  # pd to sample for the nbr per patches layer and position along tonotopic axis
    s = pd.DataFrame(np.zeros((int(height/patch_size), len(layers))))  # pd for corrected valid average orientation per layer and pos in tonotopic axis

    batch_size = 5  # batch size concerning z-axis
    for batch in range(0, int(depth / batch_size)):
        orientations = []
        dists3D = ndimage.distance_transform_edt(mask_cortex[batch * batch_size:batch * batch_size + batch_size, :, :],
                                                 sampling=[8, 1, 1], return_distances=True)

        for z in range(batch_size):
            sx = ndimage.sobel(dists3D[z], axis=0, mode='nearest')
            sy = ndimage.sobel(dists3D[z], axis=1, mode='nearest')
            sobel = np.arctan2(sy, sx) * 180 / np.pi
            sobel_smooth = gaussian_filter(sobel, sigma=2)
            orientations.append(sobel_smooth)  # angles

        for i in range(int(width / patch_size)):
            for j in range(int(height / patch_size)):
                filename = path_directionality + str(i) + '_' + str(j) + '.csv'
                path_patch = os.path.join(path, filename)
                patch = pd.read_csv(path_patch, encoding = "ISO-8859-1")
                if not header:
                    patch.columns = colnames
                patch.rename(columns={'Direction (°)': 'Direction'}, inplace=True) #�

                x = np.arange(batch * batch_size, batch * batch_size + batch_size, 1)
                for k, v in enumerate(x):
                    patch_otsu = mask_otsu[v, j * patch_size:j * patch_size + patch_size,
                                 i * patch_size:i * patch_size + patch_size]
                    cortexDepth = dists3D[k][int(j * patch_size + patch_size / 2), int(i * patch_size + patch_size / 2)]
                    key_layer = np.digitize(cortexDepth, layers, right=False)-1 # key for layer identity
                    key_tonotopy = j  # key for position in tonotopic axis
                    if 255 in patch_otsu and cortexDepth <= max_dist and np.isnan(
                            np.min(patch['Slice_' + str(v + 1)])) == False:
                        angle_cortex = orientations[k][int(j * patch_size + patch_size / 2),
                                                       int(i * patch_size + patch_size / 2)]
                        # get angle difference and rotate all orientations in patch
                        correction = 90 - angle_cortex
                        direction_corrected = patch['Direction'] - correction
                        # shift angles < -90 and > 90 degrees back into -90 to 90 range
                        patch_shifted = pd.concat([direction_corrected, patch['Slice_' + str(v + 1)]], axis=1)
                        patch_shifted['Direction'].loc[patch_shifted['Direction'] < -90] += 180
                        patch_shifted['Direction'].loc[patch_shifted['Direction'] > 90] -= 180
                        # relocate directionality values after shiften to original -90/90 interval ->take nearest value in 'Direction' and save in summary
                        summary = np.stack((np.copy(direction), np.zeros(len(direction))), axis=1)
                        for row in range(len(patch_shifted)):
                            idx = (np.abs(summary[:,0] - patch_shifted['Direction'][row])).argmin()
                            summary[idx,1] = patch_shifted['Slice_' + str(v + 1)][row]
                        #summary[:, 0] = np.radians(summary[:, 0] / math.pi)
                        stats = summary[summary[:, 1].argmax(), 0]  #get mode of directions in patch
                        s[key_layer][key_tonotopy] += stats
                        nbr[key_layer][key_tonotopy] += 1
    return s, nbr

# main
'''name_cortex = args.side + '_cortex.tif'
name_data = args.side + '_smooth2_bg95_frangi2.tif'
name_otsu = args.side + '_smooth2_bg95_otsu.tif'
path = '/media/muellerg/Data SSD/Gesine/Data/'
folder_directionality = args.side + '_frangi_' + str(args.patch_size) + '/'
name_directionality = args.side + str(args.patch_size) + '_'
path_directionality = folder_directionality + name_directionality
save_path = path + folder_directionality'''

name_cortex = 'test_C00_binMask_cortex.tif'
name_data = 'test_C03_smooth3D_bg95_frangi.tif'
name_otsu = 'test_C03_smooth3D_bg95_otsu.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/test/'
folder_directionality = 'dir_92/'
ori ='Ori'
name_directionality = 'Left92'+ori
path_directionality = folder_directionality + name_directionality
save_path = path + folder_directionality
patch_size = 92

###### for possible colnames: OrientationJ, set header to False ######
f = folder_directionality + 'Left92' + str(0) + '_' + str(0) + '.csv'
path_p = os.path.join(path, f)
p0 = pd.read_csv(path_p, encoding = "ISO-8859-1")
colnames = p0.keys()[1:][::2]
colnames = colnames.insert(0,p0.keys()[0])
#################################

start = timeit.default_timer()
stats, nbr = directionality_layer_tonotopy(name_otsu, name_cortex, path, path_directionality, patch_size,
                                           colnames, header=False, pixel=0.542)
stats.to_csv(path + '/' + folder_directionality + ori + 's.csv')
nbr.to_csv(path + '/' + folder_directionality + ori + 'nbr.csv')
stop = timeit.default_timer()
execution_time = stop - start
print('Program Executed in ' + str(round(execution_time, 2)) + ' seconds')