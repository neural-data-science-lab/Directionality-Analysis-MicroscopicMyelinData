import os
import numpy as np
import nibabel as nib
import skimage.io as io

### try to convert img to .nii
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Registration/'
static_name = 'static_AF.tif'
reference_name = 'Allen Atlas/autofluorescence_template.tif'
static_img = io.imread(os.path.join(path, static_name))
reference_img = io.imread(os.path.join(path, reference_name))

'''# https://gist.github.com/jcreinhold/dfe43f54b6dfd8bb7e9f293c0007e15d
imgs = []
for fn in range(AF.shape[0]):
    img = np.asarray(AF[fn]).astype(np.float32)
    if img.ndim != 2:
        raise Exception(f'Only 2D data supported.')
    imgs.append(img)
img = np.stack(imgs, axis=2)
nib.Nifti1Image(img,None).to_filename(os.path.join(path, 'static_AF.nii.gz')) '''

nib.Nifti1Image(static_img,None).to_filename(os.path.join(path, 'static_AF.nii.gz')) #maybe just that is sufficient
nib.Nifti1Image(reference_img,None).to_filename(os.path.join(path, 'AF_template.nii.gz'))

# load .nii
static_img = nib.load(os.path.join(path, 'static_AF.nii.gz'))
reference_img = nib.load(os.path.join(path, 'AF_template.nii.gz'))

#try out
