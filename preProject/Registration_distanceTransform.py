import skimage.io as io
import os
from scipy import ndimage
from skimage.io import imsave

path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Registration/'
mov = 'static_AF_substack(3-230)-rescaledFactor-downscaling-padded-binary.tif'
temp = 'Allen Atlas/autofluorescence_template-binary.tif'
moving = io.imread(os.path.join(path, mov))
template = io.imread(os.path.join(path, temp))
dists3D_mov = ndimage.distance_transform_edt(moving,return_distances=True)
dists3D_temp = ndimage.distance_transform_edt(template,return_distances=True)

imsave(path+"static_AF_substack(3-230)-rescaledFactor-downscaling-padded-binary-dist.tif", dists3D_mov.astype('uint16'))
imsave(path+"Allen Atlas/autofluorescence_template_binary_dist.tif", dists3D_temp.astype('uint16'))