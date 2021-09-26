import os
import h5py
import nibabel as nib
import skimage.io as io
from os.path import join as pjoin
import numpy as np
from dipy.align.imaffine import (transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D, RigidIsoScalingTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric, SSDMetric , EMMetric
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import regtools
from skimage.io import imsave

### convert img to .nii
path = 'E:/MasterThesis/'
static_name = '210719_PR010_Overview_10-55-03.tif'
moving_name = 'PR010_smaller_subset_version2.h5'
static_img = io.imread(os.path.join(path, static_name))
moving_img = h5py.File(os.path.join(path, moving_name), 'r')[u"/t00000/s00/4/cells"]
nib.Nifti1Image(static_img,None).to_filename(os.path.join(path, static_name + '.nii.gz')) #maybe just that is sufficient ToDo: define header?
nib.Nifti1Image(moving_img,None).to_filename(os.path.join(path, moving_name + '_00.nii.gz'))


### Dipy: registration: Registration of Overview on Allen Atlas
# Input: Allen Atlas, Overview (downscaled and with swaped z-axis)
# ToDo: if use .h5: downscaling probately not needed

static_data, static_affine, static_image = load_nifti(pjoin(path, '210719_PR010_Overview_10-55-03.tif.nii.gz'), return_img=True)
static = static_data
moving_data, moving_affine, moving_img = load_nifti(pjoin(path, 'PR010_smaller_subset_version2.h5_00.nii.gz'), return_img=True)
moving = moving_data

# resampling Overview dataset - transform moving towards static
identity = np.eye(4)
affine_map = AffineMap(identity, static.shape, static_affine, moving.shape, moving_affine)
resampled_moving = affine_map.transform(moving)

# COM
c_of_mass = transform_centers_of_mass(static, static_affine, moving, moving_affine)
transformed = c_of_mass.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformed, None, 1, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformed, None, 2, "Static", "Transformed", "transformed_transl_0.png")


# Registration
#RigidIsoScalingTransform (Rotation, Translation, Skalierung)
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)
level_iters = [1000, 100, 10]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
transform = TranslationTransform3D()
params0 = None
transform = RigidIsoScalingTransform3D()

starting_affine = c_of_mass.affine
rigidIso = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed = rigidIso.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformed, None, 1, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformed, None, 2, "Static", "Transformed", "transformed_transl_0.png")
save_nifti(path+'Registration_overview_pipeline1309/PR009_Overview_C00_rigidIso.nii.gz', transformed, rigidIso.affine, hdr=None)
imsave(path+"Registration_overview_pipeline1309/PR009_Overview_C00_rigidIso.tif", transformed.astype('uint16'))

#Symmetric Diffeomorphic Registration (SyN) algorithm: assume first alignment (RigidTransform3D)
metrics = [CCMetric(3), SSDMetric(3) , EMMetric(3)]
m = ['CC', 'SSD', 'EM']
level_iters = [100, 50, 10]
for i, metric in enumerate(metrics):
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(static, moving, static_affine, moving_affine, rigidIso.affine)
    warped_moving = mapping.transform(moving)
    regtools.overlay_slices(static, warped_moving, None, 0, 'Static', 'Warped moving', 'warped_moving_0.png')
    regtools.overlay_slices(static, warped_moving, None, 1, 'Static', 'Warped moving', 'warped_moving_1.png')
    regtools.overlay_slices(static, warped_moving, None, 2, 'Static', 'Warped moving', 'warped_moving_2.png')
    save_nifti(path+'Registration_overview_pipeline1309/PR009_Overview_C00_rigidIso_SyN_'+m[i]+'.nii.gz', warped_moving, mapping.codomain_grid2world, hdr=None)
    imsave(path+'Registration_overview_pipeline1309/PR009_Overview_C00_rigidIso_SyN_'+m[i]+'.tif', warped_moving.astype('uint16'))




