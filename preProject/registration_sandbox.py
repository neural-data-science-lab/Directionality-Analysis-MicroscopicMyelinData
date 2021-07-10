import os
import numpy as np
import nibabel as nib
import skimage.io as io
import matplotlib.pyplot as plt

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

nib.Nifti1Image(static_img,None).to_filename(os.path.join(path, 'static_AF.nii.gz')) #maybe just that is sufficient; ToDo: define header?
nib.Nifti1Image(reference_img,None).to_filename(os.path.join(path, 'AF_template.nii.gz'))

# load .nii
static_img = nib.load(os.path.join(path, 'static_AF.nii.gz'))
reference_img = nib.load(os.path.join(path, 'AF_template.nii.gz'))
print(static_img.header.get_zooms()[:3])

#try out registration
from os.path import join as pjoin
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

files, folder = fetch_stanford_hardi()
static_data, static_affine, static_image = load_nifti(pjoin(folder, 'HARDI150.nii.gz'), return_img=True)
static = np.squeeze(static_data)[..., 0]
static_grid2world = static_affine
files, folder2 = fetch_syn_data()
moving_data, moving_affine, moving_img = load_nifti(pjoin(folder2, 'b0.nii.gz'),return_img=True)
moving = moving_data
moving_grid2world = moving_affine
# resampling moving image
identity = np.eye(4)
affine_map = AffineMap(identity,static.shape, static_grid2world, moving.shape, moving_grid2world)
resampled = affine_map.transform(moving)
regtools.overlay_slices(static, resampled, None, 0, "Static", "Moving", "resampled_0.png")
regtools.overlay_slices(static, resampled, None, 1, "Static", "Moving", "resampled_1.png")
regtools.overlay_slices(static, resampled, None, 2, "Static", "Moving", "resampled_2.png")
# perform center of mass alignment
c_of_mass = transform_centers_of_mass(static, static_grid2world, moving, moving_grid2world)
transformed = c_of_mass.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_com_0.png")
# affine transformation
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)
level_iters = [10000, 1000, 100]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
transform = TranslationTransform3D()
params0 = None
starting_affine = c_of_mass.affine
translation = affreg.optimize(static, moving, transform, params0, static_grid2world, moving_grid2world, starting_affine=starting_affine)
transformed = translation.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_trans_0.png")


############## my data ############# todo: how to adjust for x,y,z pixel size differences and the shrink factor?
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Registration/'
static_dat, static_aff, static_img = load_nifti(pjoin(path, 'AF_template.nii.gz'), return_img=True)
static_grid = static_aff
AF_dat, AF_aff, AF_img = load_nifti(pjoin(path, 'static_AF.nii.gz'), return_img=True)
template = np.int16(AF_dat)
template_grid = AF_aff
identity = np.eye(4)
aff_map = AffineMap(identity, static_dat.shape, static_grid, template.shape, template_grid)
template_resampled = aff_map.transform(template)
regtools.overlay_slices(static_dat, template_resampled, None, 0, "Static", "Template", "resampled_0.png")
regtools.overlay_slices(static_dat, template_resampled, None, 1, "Static", "Template", "resampled_1.png")
regtools.overlay_slices(static_dat, template_resampled, None, 2, "Static", "Template", "resampled_2.png")
# perform center of mass alignment: does not work (ToDo)
COM = transform_centers_of_mass(static_dat, static_grid, template, template_grid)
template_COM = COM.transform(template)
regtools.overlay_slices(static_dat, template_COM, None, 0, "Static", "Template_COM", "template_com_0.png")
# affine registration
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)
level_iters = [10000, 1000, 100]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
transform3D = TranslationTransform3D()
params0 = None
starting_affine = AF_aff #ToDo: where to start
aff_translation = affreg.optimize(static_dat, template, transform3D, params0, static_grid, template_grid, starting_affine=starting_affine)
template_transfrom = aff_translation.transform(template)
regtools.overlay_slices(static_dat, template_transfrom, None, 0, "Static", "Template_transformed", "transformed_trans_0.png")