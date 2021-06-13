import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import timeit
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import circmean, circvar, circstd
from skimage.filters import meijering, sato, frangi, hessian
import visvis as vv
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.draw import ellipsoid
from skimage.io import imsave
from pyqtgraph.opengl import GLViewWidget, MeshData
from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem
from PyQt5.QtWidgets import QApplication
from mayavi import mlab
import scipy.spatial as spa

def normalize(image):
    min_val=np.min(image)
    max_val=np.max(image)
    return (image-min_val)/(max_val-min_val)

########## Run directionality on patches via directionality.py in fiji -> .csv files containing the histograms #########

# 1. Have a look on an exemplary histogram
def plot_histograms(filename, path):
    '''
    function to plot the histrogram of the directionality calculation from directionality.py for small patches
    data: CSV table that contains columnwise angles, histogram for slice i, fit for slice 1, ...
    '''
    filepath = os.path.join(path, filename)
    data = pd.read_csv(filepath)
    df_freq = data.iloc[:, 1::2]
    a = round(data['Direction (�)'], 2)
    labels = [str(i) for i in a]
    ax = df_freq.plot.bar(legend=False)
    ax.set_xticklabels(labels, rotation=90)

def plot_fits(filename, path):
    '''
    function to plot the Fits of the directionality calculation from directionality.py for small patches
    data: CSV table that contains columnwise angles, histogram for slice i, fit for slice 1, ...
    '''
    filepath = os.path.join(path, filename)
    data = pd.read_csv(filepath)
    df_fit = data.iloc[:, ::2]
    slices = df_fit.keys()
    slices = slices[1:]
    fig, ax = plt.subplots()
    for i in range(len(slices)):
        ax.plot(df_fit['Direction (�)'], df_fit[slices[i]], label = slices[i])
    ax.set_ylabel('Frequency')
    ax.set_title('Directionality of structures')
    ax.legend()
    fig.tight_layout()
    plt.show()

# 2. select only patches according to the binary mask: test_C03_smooth3D_bg95_otsu.tif or test_C03_smooth3D_otsu.tif
# 3. Rotate, realign orientations obtained from patches according to normal direction of brain

def rotate_directionality(name_otsu, name_cortex, path, path_directionality, patch_size = 80):
    '''
    1. extract all valid patches in the sense that based on a binary mask only those orientation patches are valid in
    which the respective mask patch is not 0;
    2. rotate the orientations from directionality calculation in order to respect cortex curvature
    (Gradient filter over distance transform)

    name_otsu:              file name of the threshold mask, test_C03_smooth3D_bg95_otsu.tif
    name_cortex:            file name of the cortex mask, test_C00_binMask_cortex.tif
    path:                   path to  files
    patch_directionality:   path to where the directionality calculation files lie
    patch_size_:            size of square patch on which the orientation was computed

    returns dictionary that contains the valid, corrected sums of orientations for all z-slices
    '''
    path_otsu = os.path.join(path, name_otsu)
    mask_otsu = io.imread(path_otsu)
    path_cortex = os.path.join(path, name_cortex)
    mask_cortex = io.imread(path_cortex)

    # dimensionality of data
    width = mask_otsu.shape[2]
    height = mask_otsu.shape[1]
    depth = mask_otsu.shape[0]

    # initialize the sum over the directionality
    file = path_directionality + str(0) + '_' + str(0) + '.csv'
    path_data = os.path.join(path, file)
    data = pd.read_csv(path_data)
    data.rename(columns={'Direction (�)': 'Direction'}, inplace=True)
    summary = pd.DataFrame(np.stack((data['Direction'], np.zeros(len(data['Direction']))), axis=1))

    d = {}
    orientations = []

    for z in range(depth): #depth
        d[str(z)] = np.copy(summary)
        # calculate gradient filter over distance transform per z-slice
        slice = mask_cortex[z]
        distances = ndimage.distance_transform_edt(slice, return_distances=True)
        sx = ndimage.sobel(distances, axis=0, mode='nearest')
        sy = ndimage.sobel(distances, axis=1, mode='nearest')
        sobel = np.arctan2(sy, sx) * 180 / np.pi
        # smooth sobel
        sobel_smooth = gaussian_filter(sobel, sigma=2)
        orientations.append(sobel_smooth)

    for i in range(int(width/patch_size)):
        for j in range(int(height/patch_size)):
            filename = path_directionality + str(i)+'_'+str(j)+'.csv'
            path_patch = os.path.join(path, filename)
            patch = pd.read_csv(path_patch)
            patch.rename(columns={'Direction (�)': 'Direction'}, inplace=True)

            for k in range(depth):
                patch_otsu = mask_otsu[k,j*patch_size:j*patch_size+patch_size,i*patch_size:i*patch_size+patch_size]
                if 255 in patch_otsu:
                    # rotate orientations according to cortex curvature
                    angle_cortex = orientations[k][int(j * patch_size + patch_size/2),
                                                   int(i * patch_size + patch_size/2)] # middle point of box
                    # get angle difference and rotate all orientations in patch
                    correction = 90-angle_cortex
                    direction_corrected = patch['Direction']-correction # substract: 90° -> 88° normal direction -> substract 90-88 = 2 degrees
                    # shift angles < -90 and > 90 degrees back into -90 to 90 range
                    patch_shifted = pd.concat([direction_corrected, patch['Slice_'+str(k+1)]], axis=1)
                    patch_shifted['Direction'].loc[patch_shifted['Direction'] < -90] += 180
                    patch_shifted['Direction'].loc[patch_shifted['Direction'] > 90] -= 180
                    for row in range(len(patch_shifted)):
                        idx = (np.abs(d[str(k)][:,0] - patch_shifted['Direction'][row])).argmin() # nearest value in directions
                        d[str(k)][idx,1] = d[str(k)][idx,1] + patch_shifted['Slice_'+str(k+1)][row]

    return d

name_otsu = 'test_C03_smooth3D_bg95_otsu.tif'
name_cortex = 'test_C00_binMask_cortex.tif'
name_data = 'test_C03_sato.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/'
path_directionality = 'test_C03_sato_dice80/directionalitySato_'

start = timeit.default_timer()
d = rotate_directionality(name_otsu, name_cortex, path, path_directionality, patch_size=80)
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(round(execution_time,2))+" seconds")

# 4. Plot preliminary results
fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
for i in range(len(d)):
    ax.plot(d[str(i)][:,0], d[str(i)][:,1], label = 'slice '+ str(i))
ax.set_ylabel('Frequency of direction', fontsize=18)
ax.set_xlabel('Directions in angluar degrees', fontsize=18)
ax.set_title('Directionality of structures in degrees', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig.tight_layout()
plt.show()

# 5. Display middle point of patches + little statistics on the image (test_C03_smooth3D_bg95.tif)
def directionality_statistics(name_otsu, name_cortex, path, path_directionality, slice, patch_size = 80):
    '''
    function to obtain a statistics from the directionality analysis

    name_otsu:              file name of the threshold mask, test_C03_smooth3D_bg95_otsu.tif
    name_cortex:            file name of the cortex mask, test_C00_binMask_cortex.tif
    path:                   path to  files
    patch_directionality:   path to where the directionality calculation files lie
    patch_size:             size of square patch on which the orientation was computed
    slice:                  2D data; z-slice which is used for statistics

    output of the function gives the i,j position of the respective patch, the angle of the correction towards the
    cortex normals and the circular statistics (mean, var, std)
    '''
    path_otsu = os.path.join(path, name_otsu)
    mask_otsu = io.imread(path_otsu)[slice]
    path_cortex = os.path.join(path, name_cortex)
    mask_cortex = io.imread(path_cortex)[slice]

    # dimensionality of data
    width = mask_otsu.shape[1]
    height = mask_otsu.shape[0]

    # initialize directionality array
    file = path_directionality + str(0) + '_' + str(0) + '.csv'
    path_data = os.path.join(path, file)
    data = pd.read_csv(path_data)
    data.rename(columns={'Direction (�)': 'Direction'}, inplace=True)
    direction = pd.DataFrame(np.stack((data['Direction'], np.zeros(len(data['Direction']))), axis=1))

    # calculate gradient filter over distance transform per z-slice
    distances = ndimage.distance_transform_edt(mask_cortex, return_distances=True)
    sx = ndimage.sobel(distances, axis=0, mode='nearest')
    sy = ndimage.sobel(distances, axis=1, mode='nearest')
    sobel = np.arctan2(sy, sx) * 180 / np.pi
    # smooth sobel
    orientations = gaussian_filter(sobel, sigma=2)

    d = []
    for i in range(int(width/patch_size)):
        for j in range(int(height/patch_size)):
            filename = path_directionality + str(i)+'_'+str(j)+'.csv'
            path_patch = os.path.join(path, filename)
            patch = pd.read_csv(path_patch)
            patch.rename(columns={'Direction (�)': 'Direction'}, inplace=True)
            patch_otsu = mask_otsu[j*patch_size:j*patch_size+patch_size,i*patch_size:i*patch_size+patch_size]
            if 255 in patch_otsu:
                # rotate orientations according to cortex curvature
                angle_cortex = orientations[int(j * patch_size + patch_size/2), int(i * patch_size + patch_size/2)] # middle point of box
                # get angle difference and rotate all orientations in patch
                correction = 90-angle_cortex
                direction_corrected = patch['Direction']-correction # substract: 90° -> 88° normal direction -> substract 90-88 = 2 degrees
                # shift angles < -90 and > 90 degrees back into -90 to 90 range
                patch_shifted = pd.concat([direction_corrected, patch['Slice_'+str(slice+1)]], axis=1)
                patch_shifted['Direction'].loc[patch_shifted['Direction'] < -90] += 180
                patch_shifted['Direction'].loc[patch_shifted['Direction'] > 90] -= 180
                summary = np.copy(direction)
                for row in range(len(patch_shifted)):
                    idx = (np.abs(summary[:,0] - patch_shifted['Direction'][row])).argmin() # nearest value in directions
                    summary[idx,1] = patch_shifted['Slice_'+str(slice+1)][row]
                summary[:,0] = np.radians(summary[:, 0] / math.pi)
                stats = np.array([i, j, correction, round(circmean(summary[:,0]*summary[:,1], high=np.pi, low = -np.pi),5),
                                  round(circvar(summary[:,0]*summary[:,1], high=np.pi, low = -np.pi),5),
                                  round(circstd(summary[:, 0] * summary[:, 1], high=np.pi, low=-np.pi),5)]) #ToDo: geht das so?
                d.append(stats)
    return d

slice = 0
patch_size = 80
stats = pd.DataFrame(directionality_statistics(name_otsu, name_cortex, path, path_directionality, slice = slice, patch_size = patch_size))

# now plot the stats onto the data image
path_data = os.path.join(path, name_data)
data = io.imread(path_data)[slice]

X = stats[0]*patch_size+patch_size/2
Y = stats[1]*patch_size+patch_size/2
angles = -stats[2]-stats[3]
U = np.cos(angles*np.pi/180)
V = np.sin(angles*np.pi/180)
fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
ax.imshow(data, cmap="gray")
ax.quiver(X, Y, U, V, color = 'green', units='xy')
#ax.plot(np.array(X), np.array(Y), 'ro', markersize = 2)


# 6. Apply frangi vesselness filter / WeKa segmentation
slice = 0
name_data = 'test_C03.tif'
path_data = os.path.join(path, name_data)
data = io.imread(path_data)

fig, ax = plt.subplots(figsize=(16, 8), dpi=180, ncols=2)
ax[0].imshow(data[slice], cmap="gray")
ax[0].set_title('Original Image 03.tiff')
ax[1].imshow(frangi(data[slice]), cmap="gray")
ax[1].set_title('Frangi filter result')
fig.tight_layout()
plt.show()

d = {}
for z in range(data.shape[0]):
    d[str(z)] = frangi(data[z], black_ridges = False)
f = np.array([normalize(d[v]) for v in d.keys()])*255
imsave(path+"test_C03_frangi.tif", f.astype('uint16'))

# 7. Plot directionality results without turning
def rotate_directionality(name_otsu, name_cortex, path, path_directionality, patch_size = 80):
    '''
    1. extract all valid patches in the sense that based on a binary mask only those orientation patches are valid in
    which the respective mask patch is not 0;
    2. rotate the orientations from directionality calculation in order to respect cortex curvature
    (Gradient filter over distance transform)

    name_otsu:              file name of the threshold mask, test_C03_smooth3D_bg95_otsu.tif
    name_cortex:            file name of the cortex mask, test_C00_binMask_cortex.tif
    path:                   path to  files
    patch_directionality:   path to where the directionality calculation files lie
    patch_size_:            size of square patch on which the orientation was computed

    returns dictionary that contains the valid, corrected sums of orientations for all z-slices
    '''
    path_otsu = os.path.join(path, name_otsu)
    mask_otsu = io.imread(path_otsu)
    path_cortex = os.path.join(path, name_cortex)
    mask_cortex = io.imread(path_cortex)

    # dimensionality of data
    width = mask_otsu.shape[2]
    height = mask_otsu.shape[1]
    depth = mask_otsu.shape[0]

    # initialize the sum over the directionality
    file = path_directionality + str(0) + '_' + str(0) + '.csv'
    path_data = os.path.join(path, file)
    data = pd.read_csv(path_data)
    data.rename(columns={'Direction (�)': 'Direction'}, inplace=True)
    summary = pd.DataFrame(np.stack((data['Direction'], np.zeros(len(data['Direction']))), axis=1))

    d = {}
    orientations = []

    for z in range(depth): #depth
        d[str(z)] = np.copy(summary)
        # calculate gradient filter over distance transform per z-slice
        slice = mask_cortex[z]
        distances = ndimage.distance_transform_edt(slice, return_distances=True)
        sx = ndimage.sobel(distances, axis=0, mode='nearest')
        sy = ndimage.sobel(distances, axis=1, mode='nearest')
        sobel = np.arctan2(sy, sx) * 180 / np.pi
        # smooth sobel
        sobel_smooth = gaussian_filter(sobel, sigma=2)
        orientations.append(sobel_smooth)

    for i in range(int(width/patch_size)):
        for j in range(int(height/patch_size)):
            filename = path_directionality + str(i)+'_'+str(j)+'.csv'
            path_patch = os.path.join(path, filename)
            patch = pd.read_csv(path_patch)
            patch.rename(columns={'Direction (�)': 'Direction'}, inplace=True)

            for k in range(depth):
                patch_otsu = mask_otsu[k,j*patch_size:j*patch_size+patch_size,i*patch_size:i*patch_size+patch_size]
                if 255 in patch_otsu:
                    # rotate orientations according to cortex curvature
                    angle_cortex = orientations[k][int(j * patch_size + patch_size/2),
                                                   int(i * patch_size + patch_size/2)] # middle point of box
                    # get angle difference and rotate all orientations in patch
                    correction = 90-angle_cortex
                    direction_corrected = patch['Direction']-correction # substract: 90° -> 88° normal direction -> substract 90-88 = 2 degrees
                    # shift angles < -90 and > 90 degrees back into -90 to 90 range
                    patch_shifted = pd.concat([direction_corrected, patch['Slice_'+str(k+1)]], axis=1)
                    patch_shifted['Direction'].loc[patch_shifted['Direction'] < -90] += 180
                    patch_shifted['Direction'].loc[patch_shifted['Direction'] > 90] -= 180
                    for row in range(len(patch_shifted)):
                        idx = (np.abs(d[str(k)][:,0] - patch_shifted['Direction'][row])).argmin() # nearest value in directions
                        d[str(k)][idx,1] = d[str(k)][idx,1] + patch_shifted['Slice_'+str(k+1)][row]

    return d
# 8. Plot heatplot etc for all z-slices

# 9. compare different cortex sides (pipeline for the huger data sets)






# code graveyard: ideas that have been withdrawn

# get fiji: somehow does not work ???
# import imagej
# ij = imagej.init(r'C:/Users/Gesine/Downloads/Fiji.app', headless=False) #'C:/Users/Gesine/Downloads/Fiji.app'
# ij.getVersion()

# Loading data from fiji procesing in order to save them without same key (for further fiji directionality calculation)
'''img_smooth = io.imread(path + 'test_C03_gaussian3D.tif')
imsave(path+"test_C03_smooth3D.tif", img_smooth)
img_bg = io.imread(path + 'test_C03_gaussian3D_bg95.tif')
imsave(path+"test_C03_smooth3D_bg95.tif", img_bg)
img_otsu = io.imread(path +'test_C03_gaussian3D_bg95_otsu.tif')
imsave(path+"test_C03_smooth3D_bg95_otsu.tif", img_otsu)
img_otsuG = io.imread(path +'test_C03_gaussian3D_otsu.tif')
imsave(path+"test_C03_smooth3D_otsu.tif", img_otsuG)'''

# Apply otsu mask on smoothed and bg image -> not helpful for directionality
'''
msk_smooth = np.copy(img_smooth)
msk_smooth[np.where(img_otsu == 0)] = 0
imsave(path+"maskBG_C03_smooth3D_otsu.tif", msk_smooth)

msk_bg = np.copy(img_bg)
msk_bg[np.where(img_otsu == 0)] = 0
imsave(path+"maskBG_C03_smooth3D_bg_otsu.tif", msk_bg)

msk2_smooth = np.copy(img_smooth)
msk2_smooth[np.where(img_otsuG == 0)] = 0
imsave(path+"maskG_C03_smooth3D_otsu.tif", msk2_smooth)

msk2_bg = np.copy(img_bg)
msk2_bg[np.where(img_otsuG == 0)] = 0
imsave(path+"maskG_C03_smooth3D_bg_otsu.tif", msk2_bg)'''

#### find cortex fails
'''contours = measure.find_contours(img_mask[0], 1000.0)
fig, ax = plt.subplots()
ax.imshow(img_mask[0], cmap=plt.cm.gray)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
#ideas: swap z and x? -> no: dim img_smooth and ellip_double stimmen überein; uint16 vs. float64
#test
ellip_base = ellipsoid(6, 10, 16, levelset=True)
ellip_double = np.concatenate((ellip_base[:-1, ...],
                               ellip_base[2:, ...]), axis=0)
v, f, n, w = measure.marching_cubes(ellip_double, 0, allow_degenerate=False, method='lewiner')

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)
ax.set_xlim(img_mask.shape[2])
ax.set_ylim(img_mask.shape[1])
ax.set_zlim(img_mask.shape[0])
plt.tight_layout()
plt.show()
'''

'''
# 3.1. idea: Marching Cubes - Start finding the cortex surface and calculate normals
binary_name = 'test_C00_binMask_cortex.tif'
binary_path = os.path.join(path, binary_name)
mask_binary = io.imread(binary_path)

verts, faces, normals, values = measure.marching_cubes(mask_binary, level=0, step_size=5,
                                                       allow_degenerate=False, method='lewiner') #, spacing=(0.542,0.542,4)

def rotate_to_surface(patch_origin,normals,verts):
    
        function to rotate orientation from directionality calculation for valid patches
        patch_origin:   get (0,0) point of patch, will be the origin of rotation
        surface_norms:  normals obtained from marching_squares
        surface_verts:  vertices obtained from marching_squares
        returns angles for rotation and rotates the directions
    

    coord_tree=spa.cKDTree(verts) #convert subset of coord_tree to a cKDTree
    idx_nearest_surface = coord_tree.query(patch_origin,k=1)[1] #find nearest neighbor to active cell in surface coords
    v_ab=normals[idx_nearest_surface]
    cellcoords=cellcoords-patch #move set so that active cell is at origin (0,0,0)

    ###find euler angles by reducing one dimension at each time, calculating the respective angle to turn around the axis at a time
    #gamm: angle between xy subvector and Y Axis -- rotation around Z axis
    v_y=np.array([0,1])#2d y vector [x,y]
    v_ab_xy=v_ab[0:2] #reduce 3d vector to XY plane
    gamm=m.acos((np.dot(v_ab_xy,v_y))/(m.sqrt(v_ab_xy[0]*v_ab_xy[0]+v_ab_xy[1]*v_ab_xy[1])*m.sqrt(v_y[0]*v_y[0]+v_y[1]*v_y[1])))    #calculate angle between v_ab_xy and y vector
    gamm=m.degrees(gamm)#angle to turn around Z axis

    #alph: angle between yz subvector and Z Axis -- rotation around X axis
    v_z=np.array([0,1])#2d z vector [y,z]
    v_ab_yz=v_ab[1:3] #reduce vector to YZ plane
    alph=m.acos((np.dot(v_ab_yz,v_z))/(m.sqrt(v_ab_yz[0]*v_ab_yz[0]+v_ab_yz[1]*v_ab_yz[1])*m.sqrt(v_z[0]*v_z[0]+v_z[1]*v_z[1])))    #calculate angle between v_ab_yz and Z vector
    alph=m.degrees(alph) #angle to turn around X axis

    #turn the dataset accordingly
    rot = R.from_euler("zyx",(gamm,0,alph),degrees=True)#initialize rotation
    coord_rotated = rot.apply(cellcoords) #rotate
    return coord_rotated, idx_nearest_surface, rot

#Plotting
app = QApplication([])
view = GLViewWidget()
mesh = MeshData(verts, faces)  # scale down - because camera is at a fixed position
mesh._vertexNormals = normals
item = GLMeshItem(meshdata=mesh, color=[1, 0, 0, 1], shader="normalColor")
view.addItem(item)
view.show()
app.exec_()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],cmap='Spectral', lw=1)
plt.show()

mlab.triangular_mesh([vert[0] for vert in verts], [vert[1] for vert in verts], [vert[2] for vert in verts], faces)
mlab.show()

vv.mesh(np.fliplr(verts), faces, normals, values)
vv.use().Run() 
'''

'''gradient = ndimage.gaussian_laplace(distances, sigma=3)
'''