import os
#import ants
import numpy as np
import nibabel as nib
import skimage.io as io
import matplotlib.pyplot as plt
from os.path import join as pjoin
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.image import load_nifti
from dipy.align.imaffine import (transform_centers_of_mass, AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D, RigidIsoScalingTransform3D)
from dipy.workflows.align import save_nifti, save_qa_metric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric, SSDMetric , EMMetric
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
import os.path
from dipy.viz import regtools
from dipy.segment.mask import median_otsu
from skimage.io import imsave


#####################################################################################################
### swap axes
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Registration/PR009_Overview_C00_downscaled/'
#static_data, static_affine, static_image = load_nifti(pjoin(path, 'AF_template.nii.gz'), return_img=True)
static_AF = io.imread(os.path.join(path, 'PR009_Overview_C00_rigidIso.tif'))
test = np.swapaxes(static_AF, 0, 2)
#nib.Nifti1Image(test,None).to_filename(os.path.join(path, 'AF_template_swapAxes.nii.gz')) #maybe just that is sufficient ToDo: define header?
imsave(path+"PR009_Overview_C00_rigidIso_02.tif", test.astype('uint16'))

######################################################################################################
### Dipy: registration
# Comments: np.int16 of data?, how to adjust for x,y,z pixel size differences and the shrink factor?
path = '/media/muellerg/Data SSD/Gesine/Data/Overview_dataset/'
static_data, static_affine, static_image = load_nifti(pjoin(path, 'AF_template.nii.gz'), return_img=True)
static = static_data
moving_data, moving_affine, moving_img = load_nifti(pjoin(path, 'PR009_Overview_C00_downscaled_swapAxes_flipZ.nii.gz'), return_img=True)
moving = moving_data

# resampling template / static
identity = np.eye(4)
affine_map = AffineMap(identity, static.shape, static_affine, moving.shape, moving_affine)
resampled_moving = affine_map.transform(moving)
regtools.overlay_slices(static, resampled_moving, None, 0, "Static", "Moving", "resampled_0.png")
regtools.overlay_slices(static, resampled_moving, None, 1, "Static", "Moving", "resampled_0.png")
regtools.overlay_slices(static, resampled_moving, None, 2, "Static", "Moving", "resampled_0.png")


# perform center of mass alignment - does not work? empty array transformed
c_of_mass = transform_centers_of_mass(static, static_affine, moving, moving_affine)
transformed = c_of_mass.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_com_0.png")
regtools.overlay_slices(static, transformed, None, 1, "Static", "Transformed", "transformed_com_0.png")
regtools.overlay_slices(static, transformed, None, 2, "Static", "Transformed", "transformed_com_0.png")


# Registration
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
translation = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed = translation.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformed, None, 1, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformed, None, 2, "Static", "Transformed", "transformed_transl_0.png")
save_nifti(path+'PR009_Overview_C00_translatTransform.nii.gz', transformed, translation.affine, hdr=None)
imsave(path+"PR009_Overview_C00_translatTransform.tif", transformed.astype('uint16'))

transform = RigidIsoScalingTransform3D()
params0 = None
starting_affine = c_of_mass.affine
rigidIso = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed = translation.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformed, None, 1, "Static", "Transformed", "transformed_transl_0.png")
regtools.overlay_slices(static, transformed, None, 2, "Static", "Transformed", "transformed_transl_0.png")
save_nifti(path+'PR009_Overview_C00_rigidIso.nii.gz', transformed, translation.affine, hdr=None)
imsave(path+"PR009_Overview_C00_rigidIso.tif", transformed.astype('uint16'))

transform = RigidTransform3D()
params0 = None
starting_affine = translation.affine
rigid = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed = rigid.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_rigid_0.png")
regtools.overlay_slices(static, transformed, None, 1, "Static", "Transformed", "transformed_rigid_0.png")
regtools.overlay_slices(static, transformed, None, 2, "Static", "Transformed", "transformed_rigid_0.png")
save_nifti(path+'PR009_Overview_C00_rigidTransform.nii.gz', transformed, rigid.affine, hdr=None)
imsave(path+"PR009_Overview_C00_rigidTransform.tif", transformed.astype('uint16'))

transform = AffineTransform3D()
params0 = None
starting_affine = c_of_mass.affine
affine = affreg.optimize(static, moving, transform, params0, static_affine, moving_affine, starting_affine=starting_affine)
transformed = affine.transform(moving)
regtools.overlay_slices(static, transformed, None, 0, "Static", "Transformed", "transformed_aff_0.png")
regtools.overlay_slices(static, transformed, None, 1, "Static", "Transformed", "transformed_aff_0.png")
regtools.overlay_slices(static, transformed, None, 2, "Static", "Transformed", "transformed_aff_0.png")
save_nifti(path+'PR009_Overview_C00_rigidIso_aff.nii.gz', transformed, affine.affine, hdr=None)
imsave(path+"PR009_Overview_C00_rigid_Iso_aff.tif", transformed.astype('uint16'))


#Symmetric Diffeomorphic Registration (SyN) algorithm: assume first alignment (RigidTransform3D)
metrics = [CCMetric(3), SSDMetric(3) , EMMetric(3)]
m = ['CC', 'SSD', 'EM']
level_iters = [100, 50, 25]
for i, metric in enumerate(metrics):
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(static, moving, static_affine, moving_affine, rigidIso.affine)
    warped_moving = mapping.transform(moving)
    regtools.overlay_slices(static, warped_moving, None, 0, 'Static', 'Warped moving', 'warped_moving_0.png')
    regtools.overlay_slices(static, warped_moving, None, 1, 'Static', 'Warped moving', 'warped_moving_1.png')
    regtools.overlay_slices(static, warped_moving, None, 2, 'Static', 'Warped moving', 'warped_moving_2.png')
    save_nifti(path+'PR009_Overview_C00_rigidIso_SyN_'+m[i]+'.nii.gz', warped_moving, mapping.codomain_grid2world, hdr=None)
    imsave(path+'PR009_Overview_C00_rigidIso_SyN_'+m[i]+'.tif', warped_moving.astype('uint16'))




###################################################################################################
### ANTsPy: registration
image = ants.image_read(ants.get_ants_data('r16'))
image2 = ants.image_read(ants.get_ants_data('r64'))
aff = ants.registration( image, image2, "Affine" )
g1 = ants.iMath_grad( image , sigma = 1, normalize=False)
g2 = ants.iMath_grad( image2 , sigma = 1, normalize=False)
ants.plot(aff['warpedmovout'])
reg1 = ants.registration( image, image2, 'SyNOnly', initial_transform=aff['fwdtransforms'][0], verbose=False )
metrics = list( )
ccMetric = ['CC', image, image2, 1.5, 4 ]
metrics.append( ccMetric )
reg2 = ants.registration( image, image2, 'SyNOnly',
    multivariate_extras = metrics, initial_transform=aff['fwdtransforms'][0] )
reg3 = ants.registration( image, image2, 'SyNOnly',
    multivariate_extras = metrics, initial_transform=aff['fwdtransforms'][0] )
print( ants.image_mutual_information( image, image2 ) )
print( ants.image_mutual_information( image, reg1['warpedmovout'] ) )
print( ants.image_mutual_information( image, reg2['warpedmovout'] ) )
print( ants.image_mutual_information( image, reg3['warpedmovout'] ) )
ants.plot(reg1['warpedmovout'])


path = '/media/muellerg/Data SSD/Gesine/Data/'
static_data = ants.image_read(path + 'autofluorescence_template-binary.tif')
moving_data = ants.image_read(path + 'static_AF_substack(3-230)-rescaledFactor-downscaling-padded-binary.tif')
affine = ants.registration( static_data, moving_data, "Affine" )
ants.plot( static_data, affine['warpedmovout'], axis=2, overlay_alpha=0.5, ncol=8, nslices=24 )


### convert img to .nii
'''path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Registration/PR009_Overview_C00_downscaled/'
static_name = 'PR009_Overview_C00_downscaled_swapAxes_flipZ.tif'
reference_name = 'AF_template_swapAxes01_flipVertically.tif'
static_img = io.imread(os.path.join(path, static_name))
reference_img = io.imread(os.path.join(path, reference_name))

# https://gist.github.com/jcreinhold/dfe43f54b6dfd8bb7e9f293c0007e15d
imgs = []
for fn in range(AF.shape[0]):
    img = np.asarray(AF[fn]).astype(np.float32)
    if img.ndim != 2:
        raise Exception(f'Only 2D data supported.')
    imgs.append(img)
img = np.stack(imgs, axis=2)
nib.Nifti1Image(img,None).to_filename(os.path.join(path, 'static_AF.nii.gz'))

nib.Nifti1Image(static_img,None).to_filename(os.path.join(path, 'PR009_Overview_C00_downscaled_swapAxes_flipZ.nii.gz')) #maybe just that is sufficient ToDo: define header?
nib.Nifti1Image(reference_img,None).to_filename(os.path.join(path, 'AF_template_swapAxes01_flipVertically.nii.gz'))
static_img = nib.load(os.path.join(path, 'static_AF.nii.gz'))
reference_img = nib.load(os.path.join(path, 'AF_template.nii.gz'))
print(static_img.header.get_zooms()[:3])'''