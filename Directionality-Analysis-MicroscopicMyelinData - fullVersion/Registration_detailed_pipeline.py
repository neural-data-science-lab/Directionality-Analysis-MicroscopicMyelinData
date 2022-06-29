import os
import h5py
import nibabel as nib
import skimage.io as io
from os.path import join as pjoin
import numpy as np
from dipy.align.imaffine import (transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (ScalingTransform3D, TranslationTransform3D, RigidTransform3D, AffineTransform3D, RigidIsoScalingTransform3D)
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric, SSDMetric , EMMetric
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import regtools
from skimage.io import imsave


### convert img to .nii
#path = '/media/muellerg/Data SSD/Gesine/Data/Registration_Overview_Detailed/Gesine Pack_1209/'
path = '/data/hdd/Philip/AJ001/'
static_name = 'C00_downscaled.tif'
moving_name = 'AJ001_stitched.ims-autofluorescence-flipVert-flipHoriz.tif' #'PR010_smaller_subset_version2.h5'
static_img = io.imread(os.path.join(path, static_name))
moving_img = io.imread(os.path.join(path, moving_name))
#moving_img = h5py.File(os.path.join(path2, moving_name), 'r')[u"/t00000/s00/4/cells"]
nib.Nifti1Image(static_img,None).to_filename(os.path.join(path, static_name + '.nii.gz')) #maybe just that is sufficient ToDo: define header?
nib.Nifti1Image(moving_img,None).to_filename(os.path.join(path, moving_name + '.nii.gz'))


### Dipy: registration: Registration of Overview on Allen Atlas
# Input: Allen Atlas, Overview (downscaled and with swaped z-axis)
path = '/data/hdd/Philip/AJ001/'
static_name = 'C00_downscaled.tif'
moving_name = 'AJ001_stitched.ims-autofluorescence-flipVert-flipHoriz.tif'
static_data, static_affine, static_image = load_nifti(pjoin(path, static_name + '.nii.gz'), return_img=True)
static = static_data
moving_data, moving_affine, moving_img = load_nifti(pjoin(path, moving_name + '.nii.gz'), return_img=True)
moving = moving_data

# simple translation
#initial_translation = np.array([[1,0,0,6.6], [0,1,0,104], [0,0,1,159], [0,0,0,1]])

# COM
c_of_mass = transform_centers_of_mass(static, static_affine, moving, moving_affine)
#c_of_mass.affine = initial_translation
transformedCOM = c_of_mass.transform(moving)
regtools.overlay_slices(static, transformedCOM, None, 0, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformedCOM, None, 1, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformedCOM, None, 2, "Static", "Transformed", "transformed_transl_0.png")
imsave(path+moving_name +'_COM.tif', transformedCOM.astype('uint16'))


# Registration
#RigidIsoScalingTransform (Rotation, Translation, Skalierung)
nbins = 32
sampling_prop = None
metric = MutualInformationMetric(nbins, sampling_prop)
level_iters = [1000, 100, 10]
sigmas = [3.0, 1.0, 0.0]
factors = [4, 2, 1]
affreg = AffineRegistration(metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors)
params0 = None


transform = RigidIsoScalingTransform3D()
starting_affine = c_of_mass.affine
rigidIso = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed_rigIso = rigidIso.transform(moving)
#regtools.overlay_slices(static, transformed_rigIso, None, 0, "Static", "Transformed", "transformed_transl_0.png")
#regtools.overlay_slices(static, transformed_rigIso, None, 1, "Static", "Transformed", "transformed_transl_0.png")
#regtools.overlay_slices(static, transformed_rigIso, None, 2, "Static", "Transformed", "transformed_transl_0.png")
save_nifti(path+moving_name + '_rigidIso.nii.gz', transformed_rigIso, rigidIso.affine, hdr=None)
imsave(path+moving_name +'_rigidIso.tif', transformed_rigIso.astype('uint16'))

transform = ScalingTransform3D()
starting_affine = rigidIso.affine
scaling = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed_scal = scaling.transform(moving)
save_nifti(path+moving_name + '_rigidIso_scaling.nii.gz', transformed_scal, scaling.affine, hdr=None)
imsave(path+moving_name +'rigidIso_scaling.tif', transformed_scal.astype('uint16'))
'''
transform = TranslationTransform3D()
starting_affine = c_of_mass.affine
translation = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed_transl = translation.transform(moving)
save_nifti(path+moving_name + '_transl.nii.gz', transformed_transl, translation.affine, hdr=None)
imsave(path+moving_name +'_transl.tif', transformed_transl.astype('uint16'))

transform = RigidTransform3D()
starting_affine = c_of_mass.affine
rigid = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed_rigid = rigid.transform(moving)
save_nifti(path+moving_name + '_rigid.nii.gz', transformed_rigid, rigid.affine, hdr=None)
imsave(path+moving_name +'_rigid.tif', transformed_rigid.astype('uint16'))'''

transform = AffineTransform3D()
starting_affine = rigidIso.affine
affine = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed_aff = affine.transform(moving)
save_nifti(path+moving_name + '_rigidIso_affine.nii.gz', transformed_aff, affine.affine, hdr=None)
imsave(path+moving_name +'rigidIso_affine.tif', transformed_aff.astype('uint16'))

#Symmetric Diffeomorphic Registration (SyN) algorithm: assume first alignment (RigidIsoTransform3D)
metrics = SSDMetric(3)
m = ['SSD']
level_iters = [100, 50, 10]
sdr = SymmetricDiffeomorphicRegistration(metrics, level_iters)
mapping = sdr.optimize(static, moving, static_affine, moving_affine, rigidIso.affine)
warped_moving = mapping.transform(moving)
regtools.overlay_slices(static, warped_moving, None, 0, 'Static', 'Warped moving', 'warped_moving_0.png')
regtools.overlay_slices(static, warped_moving, None, 1, 'Static', 'Warped moving', 'warped_moving_1.png')
regtools.overlay_slices(static, warped_moving, None, 2, 'Static', 'Warped moving', 'warped_moving_2.png')
save_nifti(path+moving_name +'_rigidIso_SyN_'+m[0]+'.nii.gz', warped_moving, mapping.codomain_grid2world, hdr=None)
imsave(path+moving_name +'_rigidIso_SyN_'+m[0]+'.tif', warped_moving.astype('uint16'))



