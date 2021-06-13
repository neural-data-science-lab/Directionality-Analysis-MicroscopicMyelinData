"""
Get an overview of the dataset:
    spacing: 0.542 x 0.542 x 4 micrometer (x,y,z)
    C00-488: Autofluoreszenz
    C01-555: HuC/D
    C02-640: TO-PRO-3
    C03-785: Myelin Basic Protein (MBP)

"""
import scipy
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.ndimage import sobel, generic_gradient_magnitude
import os
import skimage.io as io
from skimage.io import imsave
from skimage import exposure
from skimage.filters.thresholding import threshold_otsu, threshold_local, try_all_threshold
from skimage.filters import rank
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy.stats import describe
from scipy.stats import linregress


################################### Image Analysis Pipeline ####################################

# Loading and Handling Image data: short visual intro
filename = 'test_C03.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test'
filepath = os.path.join(path, filename)
outputpath = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test'
img = io.imread(filepath)
print("Array is of type:", type(img))
print("Array has shape:", img.shape) #z,y,x
print("Values are of type:", img.dtype)

# Plotting raw data: various ways
def show_plane(ax, plane, cmap="gray", title=None): # from x,y,z direction
    ax.imshow(plane, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title)
# or
# use voxel size to determine the aspect of the visualization
voxel_size_x = voxel_size_y = 0.542
voxel_size_z = 4
aspect_xz = voxel_size_z / voxel_size_x
aspect_yz = voxel_size_z / voxel_size_y

def showXYZProjection(img):
    max_z = np.max(img, axis=0)
    plt.imshow(max_z)
    plt.show()

    max_y = np.max(img, axis=1)
    plt.imshow(max_y, aspect_xz)
    plt.show()

    max_x = np.max(img, axis=2)
    plt.imshow(max_x, aspect_yz)
    plt.show()

showXYZProjection(img)


def display(im3d, step=2):  # several frames from z-dir
    _, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
    for ax, image in zip(axes.flatten(), im3d[::step]):
        ax.imshow(image, interpolation='none', cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

def plot_hist(ax, data, title=None): # histogram
    ax.hist(data.ravel(), bins=256) #.ravel() = reshape/flatten array to 1D
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    if title:
        ax.set_title(title)

'''plt.figure(figsize=(10,12)) # single image
plt.imshow(img[0,:,:], interpolation='none', cmap='gray')
plt.title('Raw Image, z = 0')
plt.show()

(n_plane, n_row, n_col) = img.shape # from x,y,z direction
_, (a, b, c) = plt.subplots(ncols=3, figsize=(15, 5))
show_plane(a, img[n_plane // 2,:,:], title=f'Plane = {n_plane // 2}')
show_plane(b, img[:, n_row // 2, :], title=f'Row = {n_row // 2}')
show_plane(c, img[:, :, n_col // 2], title=f'Column = {n_col // 2}')

display(img) # several frames from z-dir

_, ((a,b)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8)) # histogram, CDF
plot_hist(a, img, title="Original histogram")
cdf, bins = exposure.cumulative_distribution(img.ravel())
b.plot(bins, cdf, "r")
b.set_title("Original CDF")'''


# Pre-processing
## Background: enhance signal-to-noise, improve structures e.g.  Deconvolution? , Cropping,  Smoothing, Artifact, Background substraction
img8 = scipy.misc.bytescale(img.astype(float)) # convert to 8-bit -> faster computation
img_smooth8 = ndi.filters.gaussian_filter(img8, sigma=2) # Gaussian smoothing, uint8
imsave(outputpath+"test_C03_1_smooth8.tif", img_smooth8.astype(np.uint8))
img_smooth = ndi.filters.gaussian_filter(img, sigma=2) # Gaussian smoothing
imsave(outputpath+"test_C03_1_smooth16.tif", img_smooth.astype(np.uint16))

#img_equalized = exposure.equalize_hist(img) # Histogram equalization: improves contrast, enhance background noise
#display(img_equalized)

thresh = threshold_otsu(img_smooth) # Otsu thresholding: correction of technical artifacts
img_otsu = img_smooth > thresh
imsave("test_C03_1_otsu16.tif", img_otsu.astype(np.uint16))

fig = try_all_threshold(img_smooth, figsize=(15, 15), verbose=False) # Different thresholding techniques
plt.show()

# ToDo: adaptive thresholding does not work -> maybe different SE mask?

img_sobel = generic_gradient_magnitude(img_smooth, sobel) # sobel filter
imsave("test_C03_1_sobel16.tif", img_sobel.astype(np.uint16))
img_laplace = ndi.gaussian_laplace(img_smooth, sigma=2) # laplace filter
imsave("test_C03_1_laplace16.tif", img_laplace.astype(np.uint16))
# not quite helpful -> try thresholding of interesting intensity values
mask_thresh = img_smooth[np.where(img_smooth > 80)]



################## 3D image visualization: https://www.youtube.com/watch?v=lRtGqc5r6O0 ##########################
# infos about environment for reproduction (min 18:30)
### ITK: conda install itk  for medical images (CT, MRI), many filters
image = itk.imread(input_file) #is not a numpy array -> itk.array_view_from_image()
print(image.GetSpacing()) #to get gray values/spacing
# itkwidgets for 3D visualization, under development

### napari: multidimentional image viewer in python; designed for 3D
import napari
napari.gui_qt() #or %gui qt
viewer = napari.Viewer()
viewer.add_image(image)

for l in viewer.layer:  #clean viewer
    viewer.layer.remove(l)

viewer.add_image(image, scale = (aspect_xz,1,1))
# to add points e.g. peaks: from skimage import feature, feature.peak_local_max() min 16:00
viewer.add_points(np.array(peaks).T, name = 'peaks', size = 10, face_color = 'red')

