import os
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


def directionality_cortexDepth(name_otsu, name_cortex, path, path_directionality, patch_size, colnames, header,
                               nbr_cortexDepths=5, pixel=0.542):
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
    n nbr od valid patches for the normalization step later on
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
    patch0 = pd.read_csv(path_patch0, encoding = "ISO-8859-1")
    if header == False:
        patch0.columns = colnames
    patch0.rename(columns={'Direction (°)': 'Direction'}, inplace=True) #�
    direction = patch0['Direction']

    n = pd.DataFrame(np.zeros((1, nbr_cortexDepths)))  # pd to sample for the nbr of patches per cortex depth
    d = pd.DataFrame(np.column_stack((np.copy(direction), np.zeros(
        (len(direction), nbr_cortexDepths)))))  # pd for corrected valid directionality frequencies

    batch_size = 5 # batch size concerning z-axis
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
        layers = np.array([0, 60, 235, 300, 560]) / pixel # starting value of the different layers in um -> pixel value
        max_dist = 750 / pixel

        for i in range(int(width / patch_size)):
            for j in range(int(height / patch_size)):
                filename = path_directionality + str(i) + '_' + str(j) + '.csv'
                path_patch = os.path.join(path, filename)
                patch = pd.read_csv(path_patch, encoding = "ISO-8859-1")
                if header == False:
                    patch.columns = colnames
                patch.rename(columns={'Direction (°)': 'Direction'}, inplace=True)

                x = np.arange(batch * batch_size, batch * batch_size + batch_size, 1)
                for k, v in enumerate(x):
                    patch_otsu = mask_otsu[v, j * patch_size:j * patch_size + patch_size,
                                 i * patch_size:i * patch_size + patch_size]
                    cortexDepth = dists3D[k][int(j * patch_size + patch_size / 2), int(i * patch_size + patch_size / 2)]
                    key = np.digitize(cortexDepth, layers, right=False)
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
                        # relocate directionality values after shifting to original -90/90 (Direction) interval ->take nearest value in 'Direction' and save in value to correct entry (idx)
                        for row in range(len(patch_shifted)):
                            idx = (np.abs(d[0] - patch_shifted['Direction'][row])).argmin()
                            d[key][idx] += patch_shifted['Slice_' + str(v + 1)][row]
                        n[key - 1] += 1
    return d, n


# main
'''name_cortex = args.side + '_cortex.tif'
name_data = args.side + '_smooth2_bg95_frangi2.tif'
name_otsu = args.side + '_smooth2_bg95_otsu.tif'
path = '/media/muellerg/Data SSD/Gesine/Data/'
folder_directionality = args.side + '_frangi_' + str(args.patch_size) + '/'
name_directionality = args.side + str(args.patch_size) + '_'
path_directionality = folder_directionality + name_directionality
cortexDepths = 5
save_path = path + folder_directionality'''

name_cortex = 'test_C00_binMask_cortex.tif'
name_data = 'test_C03_smooth3D_bg95_frangi.tif'
name_otsu = 'test_C03_smooth3D_bg95_otsu.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/test/'
folder_directionality = 'dir_92/'
name_directionality = 'Left92Ori'
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
'''corrected, nbr = directionality_cortexDepth(name_otsu, name_cortex, path, path_directionality, args.patch_size,
                                            nbr_cortexDepths=5, pixel=0.542)'''
corrected, nbr = directionality_cortexDepth(name_otsu, name_cortex, path, path_directionality, patch_size, colnames, header = False,
                                            nbr_cortexDepths=5, pixel=0.542)
corrected.to_csv(path + '/' + folder_directionality + 'd.csv')
nbr.to_csv(path + '/' + folder_directionality + 'n.csv')
stop = timeit.default_timer()
execution_time = stop - start
print('Program Executed in ' + str(round(execution_time, 2)) + ' seconds')