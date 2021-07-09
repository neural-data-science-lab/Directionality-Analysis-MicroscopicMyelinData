import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import tifffile as tf
from ipywidgets import interact, fixed
from IPython.display import clear_output
import h5py
from skimage.transform import resize
import numpy as np
import math
from scipy import ndimage as ndi

plot = True #plot gradient descent and comparison of finished transformations
step1 = True #fixed transformation. Reference atlas -> Downscaled microscopy image
step2 = True #affine transformation Reference atlas -> Downscaled microscopy image
step3 = True #affine transformation of pre-registered image to downscaled high resolution image
step4 = False #upscaling of atlas back to full-size dataset size

output_dir = r"C:\Data\AJ003\results" #dump of all intermediate and resulting images
upscaled_atlas_path = r"C:\Data\AJ003\results\atlas.hdf5"
overview_img_path = r"C:\Data\AJ003\AJ003_overview_downscaled_further_kept.tif" #Overview microscopy image.
hires_img_path = r"C:\Data\AJ003\AJ003_highres_downscaled_z_2.tif" #step3: fixed img, downscaled
reference_img_path = r"C:\Data\AJ003\autofluorescence_template.tif" #AF template image.
atlas_img_path = r"C:\Data\AJ003\annotation_atlas.tif" #untransformed atlas path

# output_dir = r"F:\Philip\AJ001\Registration\results2" #dump of all intermediate and resulting images
# upscaled_atlas_path = r"F:\Philip\AJ001\Registration\atlas.hdf5"
# overview_img_path = r"F:\Philip\AJ001\Registration\C00_overview_downscaled.tif" #Overview microscopy image.
# hires_img_path = r"F:\Philip\AJ001\Registration\C00_detail_downscaled_2_rotated.tif" #step3: fixed img, not downscaled yet
# reference_img_path = r"F:\Philip\AJ001\Registration\autofluorescence_template.tif" #AF template image.
# atlas_img_path = r"F:\Philip\AJ001\Registration\annotation_atlas.tif" #untransformed atlas path

shape_upscaled = (450,6797,14433) #shape of full-size image
batch_shape   = (20,20,20)   #batch shape (Z,Y,X) of batches processed for upscaling the atlas

#factors of how much hires_img_path image is scaled down at the end of the process (prior downscaling x downscale_factor)
z_factor = 1
y_factor = 24
x_factor = 24

#find prime factorials for a given number
def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    return primfac

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.    
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

def sobel(im,sigma1,sigma2):
    #pre-gauss to eliminate stripes
    im=ndi.gaussian_filter(im,sigma1)

    # Compute sobel filter along all three axes
    edges1 = ndi.sobel(im, axis=0)
    edges2 = ndi.sobel(im, axis=1)
    edges3 = ndi.sobel(im, axis=2)

    # Average images
    edges_sum = (edges1+edges2+edges3)/3
    edges_sum[edges_sum<0] = edges_sum[edges_sum<0]*(-1)

    #gauss img
    edges_sum = ndi.gaussian_filter(edges_sum,sigma2)
    
    return edges_sum

if step1 == True: ### rigid alignment - Reference atlas to overview microscopy image
    print("starting rigid alignment (#1)")

    #load images
    moving_image = tf.imread(reference_img_path)
    fixed_image = tf.imread(overview_img_path)
    atlas_raw = tf.imread(atlas_img_path)

    #convert images to sitk format
    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)
    atlas_raw = sitk.GetImageFromArray(atlas_raw)

    # initialize transformation
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                        moving_image, 
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    moving_rigid_transformed = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # registration routine
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=80)#default:100 // 2500
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.04)#default: 0.01 // 0.15

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings. defaults: learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10 // 20
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.01, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [12,4])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3,2])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    if plot == True:    
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                sitk.Cast(moving_image, sitk.sitkFloat32))

    # apply transform to moving image and atlas image
    moving_rigid_transformed = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    atlas_transformed = sitk.Resample(atlas_raw, fixed_image, final_transform, sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID())

    # save results to file
    tf.imsave(output_dir + r"\step1_moving_image.tif", sitk.GetArrayFromImage(moving_image))
    tf.imsave(output_dir + r"\step1_moving_transformed_image.tif", sitk.GetArrayFromImage(moving_rigid_transformed))
    tf.imsave(output_dir + r"\step1_reference_image.tif", sitk.GetArrayFromImage(fixed_image))
    tf.imsave(output_dir + r"\step1_atlas_image.tif", sitk.GetArrayFromImage(atlas_transformed))

    sitk.WriteTransform(final_transform, output_dir + r"\step1_rigid_transform.tfm")

if step2 == True: ### affine transform - registers pre-aligned atlas AF image to microscopy img

    print ("starting affine transform alignment (#2)")

    #load reference AF image
    fixed_image = tf.imread(overview_img_path)
    moving_image = moving_rigid_transformed

    #convert to sitk format
    fixed_image = sitk.GetImageFromArray(fixed_image)

    # initialize affine transformation
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                        moving_image, 
                                                        sitk.AffineTransform(3), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    moving_rigid_transformed = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # registration routine

    # initialize
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)#default:100
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.02)#default: 0.01

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings. defaults: learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10 
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.7,
                                                    numberOfIterations=150,#100
                                                    convergenceMinimumValue=1e-6,
                                                    convergenceWindowSize=5)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    if plot == True:    
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                sitk.Cast(moving_image, sitk.sitkFloat32))

    # apply rigid transform to image
    moving_affine_transformed = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    atlas_transformed = sitk.Resample(atlas_transformed, fixed_image, final_transform, sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID())

    # save results to file
    tf.imsave(output_dir + r"\step2_moving_image.tif", sitk.GetArrayFromImage(moving_image))
    tf.imsave(output_dir + r"\step2_moving_transformed_image.tif", sitk.GetArrayFromImage(moving_affine_transformed))
    tf.imsave(output_dir + r"\step2_reference_image.tif", sitk.GetArrayFromImage(fixed_image))
    tf.imsave(output_dir + r"\step2_atlas_image.tif", sitk.GetArrayFromImage(atlas_transformed))
    sitk.WriteTransform(final_transform, output_dir + r"\step2_affine_transform.tfm")

if step3 == True: ### affine alignment of registered overview image to downscaled high resolution image
    print("starting affine alignment (#3)")
    print("loading high resolution image")
    hires_img = tf.imread(hires_img_path) #step3:fixed img, not downscaled or rotated yet

    output_shape = np.array(hires_img.shape)
    print ("downscaling..")

    downscaled_hiresimg = resize(hires_img,
                                 output_shape=tuple(output_shape),
                                 anti_aliasing=False,
                                 mode="reflect",
                                 order=0,
                                 preserve_range=True,)

    moving_image = sitk.GetArrayFromImage(fixed_image)

    print ("detecting edges")
    #sobel filter both low-res and hi-res microscopy image
    downscaled_hiresimg = sobel(downscaled_hiresimg.astype("float32"),sigma1=0,sigma2=1)
    moving_image_edge = sobel(moving_image.astype("float32"),sigma1=8,sigma2=3)

    #rotate atlas and moving image (microscopy img from last step) to right orientation
    rotated_mov = np.rot90(moving_image_edge,k=3,axes=(0,1))
    rotated_atl = np.rot90(sitk.GetArrayFromImage(atlas_transformed),k=3,axes=(0,1))

    #convert atlas and moving image back to sitk format
    rotated_mov = sitk.GetImageFromArray(rotated_mov.astype("float32"))
    rotated_atl = sitk.GetImageFromArray(rotated_atl)

    #load images
    moving_image = rotated_mov #fixed img from last step is now moving img
    fixed_image = downscaled_hiresimg.astype("float32")
    atlas_raw = rotated_atl

    #convert images to sitk format
    fixed_image = sitk.GetImageFromArray(fixed_image) #other two are already in sitk format
    print ("starting step3 alignment")
    # initialize transformation
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                          moving_image, 
                                                          sitk.AffineTransform(3),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
  
    moving_rigid_transformed = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # registration routine
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100) #default:100 // 2500
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)#.REGULAR
    registration_method.SetMetricSamplingPercentage(0.15) #default: 0.01 // 0.15 #0.3

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings. defaults: learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10 // 20  //40
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=400, convergenceMinimumValue=1e-6, convergenceWindowSize=40)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    #for AJ001:learningrate 0.04

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors =   [4,2])#8,4,2])#16,8,4,2,1
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4,3])#4,3,2])#54310
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    if plot == True:    
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    # apply transform to moving image and atlas image
    moving_rigid_transformed = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    atlas_transformed = sitk.Resample(atlas_raw, fixed_image, final_transform, sitk.sitkNearestNeighbor, 0.0, moving_image.GetPixelID())

    # save results to file
    tf.imsave(output_dir + r"\step3_moving_image.tif", sitk.GetArrayFromImage(moving_image))
    tf.imsave(output_dir + r"\step3_moving_transformed_image.tif", sitk.GetArrayFromImage(moving_rigid_transformed))
    tf.imsave(output_dir + r"\step3_reference_image.tif", sitk.GetArrayFromImage(fixed_image))
    tf.imsave(output_dir + r"\step3_atlas_image.tif", sitk.GetArrayFromImage(atlas_transformed))

    sitk.WriteTransform(final_transform, output_dir + r"\step3_rigid_transform.tfm")

    print ("Registration steps done")

if step4 == True: ### upscaling of atlas to original image size
        
    atlas = sitk.GetArrayFromImage(atlas_transformed).astype("uint16") #atlas image

    f = h5py.File(upscaled_atlas_path,"w") #new file for upscaled atlas

    f.create_dataset("atlas", shape=atlas.shape, dtype="uint16", chunks=True, maxshape=(None,None,None))
    f.create_dataset("atlas_upscaled", shape=shape_upscaled, dtype="uint16", chunks=True, maxshape=(None,None,None))

    f[u"atlas"][:] = atlas #insert atlas into h5

    ### initial coordinates and batch size
    z_start = 0 
    z_end   = batch_shape[0]
    y_start = 0
    y_end   = batch_shape[1]
    x_start = 0
    x_end   = batch_shape[2]

    output_shape = z_end*z_factor-z_start*z_factor,\
                   y_end*y_factor-y_start*y_factor,\
                   x_end*x_factor-x_start*x_factor

    print("upscaling..")

    n_upscales = atlas.shape[0]*atlas.shape[1]*atlas.shape[2]//batch_shape[0]*batch_shape[1]*batch_shape[2]
    n_iter = 0
    percent = 0

    while x_end <= atlas.shape[2]:

        n_iter = n_iter +1

        if (n_iter in np.linspace(0,n_upscales,100,dtype="uint32")) == True:
            percent = percent+1
            print(str(percent) + "%")

        # print (z_start,y_start,x_start)
        
        #upscaling section of the image with nearest-neighbour interpolation (spline order=0)
        upscaled_section = resize(f[u"atlas"][z_start:z_end,\
                                            y_start:y_end,\
                                            x_start:x_end],
                                            output_shape=output_shape,
                                            anti_aliasing=False,
                                            order=0,
                                            mode="reflect",
                                            preserve_range=True,)

        f[u"atlas_upscaled"][int(z_start*z_factor):int(z_start*z_factor+upscaled_section.shape[0]),\
                             int(y_start*y_factor):int(y_start*y_factor+upscaled_section.shape[1]),\
                             int(x_start*x_factor):int(x_start*x_factor+upscaled_section.shape[2])] = upscaled_section

        z_start = z_start + batch_shape[0]
        z_end   = z_start + batch_shape[0]
        
        # if z_end exceeds img shape: move y_start (and reset z_start)
        if z_end > atlas.shape[0]:
            y_start = y_start + batch_shape[1]
            y_end   = y_start + batch_shape[1]
            z_start = 0
            z_end   = batch_shape[0]
        
        # if z_end AND y_end exceed img shape: move x_start (and reset y_start and z_start)
        if y_end > atlas.shape[1]:
            x_start = x_start + batch_shape[2]
            x_end   = x_start + batch_shape[2]
            z_start = 0
            z_end   = batch_shape[0]
            y_start = 0
            y_end   = batch_shape[1]
            
    plt.imshow(f[u"atlas_upscaled"][200])
    plt.show()
 
print("All done!")


### code graveyard

    # # show aligned images
    # plt.subplots(2,2,figsize=(12,10))

    # half_z = int(sitk.GetArrayFromImage(fixed_image).shape[0]//2)

    # # Draw the fixed image in the first subplot.
    # plt.subplot(2,2,1)
    # plt.imshow(sitk.GetArrayFromImage(fixed_image)[half_z,:,:],cmap="gray")
    # plt.title('fixed image')
    # plt.axis('off')

    # # Draw the moving image in the second subplot.
    # plt.subplot(2,2,2)
    # plt.imshow(sitk.GetArrayFromImage(moving_image)[half_z,:,:],cmap="gray")
    # plt.title('moving image before alignment')
    # plt.axis('off')

    # # Draw the moving image in the third subplot.
    # plt.subplot(2,2,3)
    # plt.imshow(sitk.GetArrayFromImage(moving_rigid_transformed)[half_z,:,:],cmap="gray")
    # plt.title('moving image after alignment')
    # plt.axis('off')

    # # add both images together and show in fourth subplot.
    # compound_img = sitk.GetArrayFromImage(moving_rigid_transformed) + sitk.GetArrayFromImage(fixed_image)
    # plt.subplot(2,2,4)
    # plt.imshow(compound_img[half_z,:,:],cmap="gray")
    # plt.title('moving image + fixed image')
    # plt.axis('off')

    # plt.show()
