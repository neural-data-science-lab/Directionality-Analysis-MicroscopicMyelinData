# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:05:02 2018
@author: Philip Ruthig, PhD Student at MPI for Cognitive and Brain Sciences, Leipzig University

todo:
- use chunk generator instead of manually generating chunks
"""

# In[Initiation of variables, packages and functions]
import os
import time
import datetime
import h5py
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig
import math as m
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean
from skimage.feature import peak_local_max
import skimage.morphology as morph
import tifffile as tf


def closeall():
    results.close()
    # img_file.close()
    return None


def gaborkernel(edge_length, sigma, freq, phase, radius, z_scale_factor):
    '''
    Returns two gabor kernels (real and imaginary) for spheroid detection. Can be convolved with an image (using scipy.signal.fftconvolve)

    edge_length: edge length of the kernel array
    sigma: sigma of gaussian part of the gabor kernel
    freq: frequency of the complex plane wave
    phase: phase displacement in pixels
    radius: radius of how much the gaussian wave is displaced from the origin
    z_scale_factor: factor of how much z axis is compressed. 1 for isotropic data
    '''
    if edge_length % 2 == 0:
        edge_length = edge_length + 1

    size_z = np.arange(0, edge_length, 1)
    size_y = np.arange(0, edge_length, 1)
    size_x = np.arange(0, edge_length, 1)

    z, y, x = np.meshgrid(size_z, size_y, size_x)
    z, y, x = z - (len(size_z) // 2), y - (len(size_y) // 2), x - (len(size_x) // 2)
    y = y * z_scale_factor

    A = (2 * m.pi * sigma ** 2)
    r = np.sqrt(np.power(z, 2) + np.power(y, 2) + np.power(x, 2))

    kernel_real = (1 / A) * np.exp(-1 * m.pi * ((np.power((r - radius), 2)) / sigma ** 2)) * (
        np.cos((freq * (r - radius) * 2 * m.pi + phase)))
    kernel_imag = (1 / A) * np.exp(-1 * m.pi * ((np.power((r - radius), 2)) / sigma ** 2)) * (
        np.sin((freq * (r - radius) * 2 * m.pi + phase)))

    # inverting kernels
    kernel_real = kernel_real * (-1)
    kernel_imag = kernel_imag * (-1)

    return kernel_real, kernel_imag


def cell_centre_distribution(bool_input, reach, ):
    '''
    Computes a mean distribution of cells in a given boolean array in a given edge length cube. Cycles through all TRUE pixels
    and checks the surroundings in radius "reach".

    bool_input: boolean numpy array input
    reach: edge length of the cube
    '''

    struct = np.zeros((3, 3, 3))
    struct[1, 1, 1] = 1

    centres_labeled, n_cells_labeled = ndi.label(bool_input, structure=struct)

    invalid_area_mask = np.ones_like(bool_input, dtype="bool")  # define valid part of data - exclude outer rim
    invalid_area_mask[0:reach // 2, :, :] = False
    invalid_area_mask[-reach // 2:, :, :] = False
    invalid_area_mask[:, -reach // 2:, :] = False
    invalid_area_mask[:, 0:reach // 2, :] = False
    invalid_area_mask[:, :, -reach // 2:] = False
    invalid_area_mask[:, :, 0:reach // 2] = False

    for i in range(n_cells_labeled):  # iterate over all cells
        if i == 0:  # initialize analysis
            resultarray = np.zeros((reach, reach, reach))  # initialize results array
            n_valid_cells = 0
            n_invalid_cells = 0
            continue

        if i % 500 == 0:  # timekeeping
            print("running analysis on cell number " + str(i) + " of " + str(n_cells_labeled))

        clocz, clocy, clocx = np.nonzero(centres_labeled == i)  # get active cell coordinates

        if invalid_area_mask[clocz[0], clocy[0], clocx[0]] == True:
            n_valid_cells += 1
            tmpdata = bool_input[clocz[0] - reach // 2:clocz[0] + reach // 2,
                      clocy[0] - reach // 2:clocy[0] + reach // 2, clocx[0] - reach // 2:clocx[0] + reach // 2]
        else:
            n_invalid_cells += 1
            continue

        resultarray = resultarray + tmpdata  # add tmp data to complete results array

    resultarray[reach // 2, reach // 2, reach // 2] = 0  # delete reference cell
    return resultarray


### general script options
plot = False  # Toggle plotting of preliminary 2d slices
colormap = "gray"  # colormap for 2d plots
size = 8  # size of displayed plots
save = True  # Toggle saving of results
compression = None  # lossless compression (gzip) of results h5 file, between 1-9. Higher = more compression
cell_dist = True  # toggle if cell centre distribution should be computed or not (takes a while)
shutdown = False  # Toggle shutdown after script is finished
open_path = r"C:\Users\Gesine\Documents\Studium\MasterCMS\MasterThesis\Analyse_Directionality\Testdatensatz-0504"  # path to folder with images (in h5 format if h5 == True)

### tuple of image names to be processed
img_names = (r"\C01.tif",)  # images with structures to be annotated

save_path = r"C:\Users\Gesine\Documents\Studium\MasterCMS\MasterThesis\Analyse_Directionality\Testdatensatz-0504\coluumn_sandbox"  # path to results folder
checkprogress = 5  # check progress of script every x batches
batch_shape = (50, 512, 432)  # shape (Z,Y,X) of the batches being processed. Larger is faster but also requires more RAM. Use even numbers
h5 = False  # is the input image h5? no = tif

### variables that will need tweaking depending on the image being processed
gauss_sigma = 0.7  # sigma for gaussian blurred image

hucd = True  # True means HuCD data. False means Topro data.

### gabor filter variables. need to be adapted to cell size, data & quality
if hucd == True:
    edge = 10#40  # gabor kernel edge length (x=y=z)
    gabor_freq = 1 / 10  # frequency of wave component
    gabor_phase = 3.7  # wave component offset
    gabor_sigma = 8  # gaussian deviation #7
    gabor_radius = 9  # donut radius       #10
    z_factor = 7  # factor of how much the z axis is compressed in microscopy data. 1 for isotropic data
    tissue_thresh = 315#530  # intensity threshold to check cells are in tissue
    r_maxima = 6  # radius of footprint for detecting local maxima in the gabor filtered image #6
    min_distance = 4  # min distance between local maxima #4

if hucd == False:
    edge = 40  # gabor kernel edge length (x=y=z)
    gabor_freq = 1 / 10  # frequency of wave component
    gabor_phase = 3.7  # wave component offset
    gabor_sigma = 8  # gaussian deviation #7
    gabor_radius = 4.5  # donut radius       #10
    z_factor = 7  # factor of how much the z axis is compressed in microscopy data. 1 for isotropic data
    tissue_thresh = 700  # intensity threshold between background max intensity and tissue min intensity. To check if detected cells are in tissue
    r_maxima = 6  # radius of footprint for detecting local maxima in the gabor filtered image #6
    min_distance = 4  # min distance between local maxima #4

### data analysis variables
reach = 50  # number of surrounding pixels taken into account for cell distribution analysis

### initializing variables and structuring elements
footprint_maxima = morph.ball(
    r_maxima)  # structuring element for detecting local maxima in gabor annulus filtered image

for img_name in img_names:  # execute for all images in list. Must be inside directory of open_path
    n_iter = 0  # number of batches processed
    n_cells = 0  # total number of identified cells
    print("starting analysis for " + img_name)

    ### initialize variables for timekeeping and timestamping data
    start_time = time.time()
    timestamp = datetime.datetime.fromtimestamp(start_time).strftime(r'%Y%m%d_%H.%M')

    # open raw image file and dataset
    if h5 == True:
        img_file = h5py.File(open_path + img_name, "r")
        img_shape = img_file[u'/t0/channel0'].shape

    if h5 == False:  # open tif
        img_file = tf.imread(open_path + img_name)
        img_shape = img_file.shape

    # approximate batches to be computed
    n_approx_batches = (img_shape[0] * img_shape[1] * img_shape[2]) / (
                (batch_shape[0] - edge) * (batch_shape[1] - edge) * (batch_shape[2] - edge))
    print("approximate number of batches to be calculated: " + str(int(n_approx_batches)))

    if save == True:
        # initialize empty h5 file for results
        results = h5py.File(save_path + r"\results_" + str(img_name[1:]) + "_" + str(timestamp) + r".h5", "w")
        results.create_dataset(u"gaussed_image",
                               dtype="uint16",
                               chunks=True,
                               compression=compression,
                               shape=(img_shape))

    print(str(int(time.time() - start_time)) + " seconds for loading h5 file.")

    # In[Main Sequence]
    # main sequence starts here. Iterates through z, then through y (and z) and then through x (and y and z) axis batch-wise.
    # The process is finished when the x-coordinates of the batch exceed the x edge length of the original Image.

    ### initial coordinates and batch size
    z_start = 0
    z_end = batch_shape[0]
    y_start = 0
    y_end = batch_shape[1]
    x_start = 0
    x_end = batch_shape[2]

    while x_end <= img_shape[2]:
        # while n_iter<2:
        # timekeeping
        if n_iter > 0 and n_iter % checkprogress == 0:  # print mid-analysis results every x iterations
            print("number of total cells found so far: " + str(n_cells))
            print("Elapsed time: " + str(int((time.time() - start_time) / 60)) + " minutes")
            approx_rest_time = int(((time.time() - start_time) / n_iter) * (n_approx_batches - n_iter))
            print("Approximate time until analysis is finished: " + str(approx_rest_time / 60) + " minutes")

        n_iter += 1
        print("start processing of batch #" + str(n_iter))

        if h5 == True:
            current_img = img_file[u'/t0/channel0'][z_start:z_end, y_start:y_end, x_start:x_end]
        if h5 == False:
            current_img = img_file[z_start:z_end, y_start:y_end, x_start:x_end]

        current_img = ndi.filters.gaussian_filter(current_img, sigma=gauss_sigma)

        # Preprocessing of Image
        img = current_img  # renaming to img
        img = img.astype("float32")

        # Gaussian Blur
        # applying a moderate gaussian filter to create a foreground image
        gaussed_img = ndi.filters.gaussian_filter(img, sigma=gauss_sigma)

        if plot == True:
            print("printing gauss filtered image")
            plt.figure(figsize=(size, size))
            plt.imshow(gaussed_img[0, :, :], interpolation='none', cmap=colormap)
            plt.colorbar()
            plt.show()

        # creating filter kernel
        gabor_kernel_real, gabor_kernel_imag = gaborkernel(
            edge_length=edge,
            sigma=gabor_sigma,
            freq=gabor_freq,
            radius=gabor_radius,
            z_scale_factor=z_factor,
            phase=gabor_phase
        )

        # applying gabor filter on valid part of the image (returning an image of the same shape)
        # gabor_edge_length//2 on every end of every axis has to be cropped when saving
        gaborimg_real = sig.fftconvolve(img, gabor_kernel_real, mode="same")

        if hucd == False:
            gaborimg_real = gaborimg_real * (-1)  # why does my image invert ??

        # adjusting img intensity
        gaborimg_real = gaborimg_real * gaussed_img
        gaborimg_real = gaborimg_real.astype("float32")

        if plot == True:
            print("printing uncropped gabor image")
            plt.figure(figsize=(size, size))
            plt.imshow(gaborimg_real[edge // 2 + 1, :, :], cmap=colormap)
            plt.colorbar()
            plt.show()

            # marking local maxima
        centers = peak_local_max(
            image=gaborimg_real,
            min_distance=min_distance,
            indices=False,
            footprint=footprint_maxima,
            exclude_border=0,
        ) # centers = peak_local_max(gaborimg_real, min_distance, footprint=footprint_maxima, exclude_border=0, indices=False)

        # threshold centers according to tissue background intensity
        centers[gaussed_img < tissue_thresh] = 0  #? only one gaussed_img < tissue_thresh == False

        # crop center coordinates image to valid part
        centers = centers[edge // 2:-edge // 2, edge // 2:-edge // 2, edge // 2:-edge // 2]

        # label and count cell centers
        cellcenters_labeled, n_cells_batch = ndi.label(centers)

        # insert cell centers at max intensity into gaussed image
        gaussed_img_cellcenters = np.copy(gaussed_img[edge // 2:-edge // 2, edge // 2:-edge // 2, edge // 2:-edge // 2])
        gaussed_img_cellcenters[centers == True] = 65535

        # insert cell centers at max intensity into gabor filtered image
        gaborimg_cellcenters = np.copy(gaborimg_real[edge // 2:-edge // 2, edge // 2:-edge // 2, edge // 2:-edge // 2])
        gaborimg_cellcenters[centers == True] = 65535

        # crop gaussed image and gabor image to valid analyzed part
        gaborimg_real = gaborimg_real[edge // 2:-edge // 2, edge // 2:-edge // 2, edge // 2:-edge // 2]
        gaussed_img = gaussed_img[edge // 2:-edge // 2, edge // 2:-edge // 2, edge // 2:-edge // 2]

        # create datasets for cellcenters, gabor image and gaussimg/gaborimg + cellcenters
        if n_iter == 1 and save == True:
            results.create_dataset(u"cellcenters",
                                   dtype="uint8",
                                   chunks=True,
                                   compression=compression,
                                   shape=(img_shape))

            results.create_dataset(u"gabor_cellcenters",
                                   dtype="float32",
                                   chunks=True,
                                   compression=compression,
                                   shape=(img_shape))

            results.create_dataset(u"gaussed_cellcenters",
                                   dtype="uint16",
                                   chunks=True,
                                   compression=compression,
                                   shape=(img_shape))

            results.create_dataset(u"gabor_image",
                                   dtype="float32",
                                   chunks=True,
                                   compression=compression,
                                   shape=(img_shape))

            results.create_dataset(u"result_array",
                                   dtype="uint16",
                                   chunks=True,
                                   compression=compression,
                                   shape=(reach, reach, reach))

            results[u"gabor_filter"] = gabor_kernel_real

        if save == True:  # insert batches into h5 datasets
            results[u"gaussed_image"][(z_start) + edge // 2:(z_end) - edge // 2,
            (y_start) + edge // 2:(y_end) - edge // 2, (x_start) + edge // 2:(x_end) - edge // 2] = gaussed_img
            results[u"gabor_image"][(z_start) + edge // 2:(z_end) - edge // 2,
            (y_start) + edge // 2:(y_end) - edge // 2, (x_start) + edge // 2:(x_end) - edge // 2] = gaborimg_real
            results[u"cellcenters"][(z_start) + edge // 2:(z_end) - edge // 2,
            (y_start) + edge // 2:(y_end) - edge // 2, (x_start) + edge // 2:(x_end) - edge // 2] = centers
            results[u"gaussed_cellcenters"][(z_start) + edge // 2:(z_end) - edge // 2,
            (y_start) + edge // 2:(y_end) - edge // 2,
            (x_start) + edge // 2:(x_end) - edge // 2] = gaussed_img_cellcenters
            results[u"gabor_cellcenters"][(z_start) + edge // 2:(z_end) - edge // 2,
            (y_start) + edge // 2:(y_end) - edge // 2, (x_start) + edge // 2:(x_end) - edge // 2] = gaborimg_cellcenters

        n_cells = n_cells + n_cells_batch
        print("cells found this batch: " + str(n_cells_batch))
        z_start = z_start + np.shape(current_img)[0] - 2 * edge
        z_end = z_start + batch_shape[0]

        # if z_end exceeds img shape: move y_start (and reset z_start)
        if z_end >= img_shape[0]:
            y_start = y_start + np.shape(current_img)[1] - 2 * edge
            y_end = y_start + batch_shape[1]
            z_start = 0
            z_end = batch_shape[0]

        # if z_end AND y_end exceed img shape: move x_start (and reset y_start and z_start)
        if y_end >= img_shape[1]:
            x_start = x_start + np.shape(current_img)[2] - 2 * edge
            x_end = x_start + batch_shape[2]
            z_start = 0
            z_end = batch_shape[0]
            y_start = 0
            y_end = batch_shape[1]

    if cell_dist == True:
        # In[Cell centre location Analysis]
        # calling cell_centre_distribution function from toolbox
        print("starting analysis of cell centers")
        resultarray = cell_centre_distribution(bool_input=results[u"cellcenters"], reach=reach)

        if save == True:
            results[u"result_array"][:, :, :] = resultarray.astype("uint16")

        if plot == True:
            plt.figure(figsize=(8, 8))
            plt.imshow(resultarray[resultarray.shape[0] // 2, :, :], cmap="inferno")
            plt.colorbar()
            plt.show()

    if cell_dist == False:
        resultarray = np.zeros_like(results[u"cellcenters"])  # placeholder empty array

    # In[Saving of the resulting images and used variables]

    if save == True:
        with open(save_path + r"\data" + timestamp + r".txt", "w") as variables:
            variables.write("name of image = " + str(img_name) + "\n")
            variables.write(
                "batch shape = " + str(batch_shape[0]) + "," + str(batch_shape[1]) + "," + str(batch_shape[2]) + "\n")
            variables.write("gaussian sigma = " + str(gauss_sigma) + "\n")
            variables.write("gabor edge length = " + str(edge) + "\n")
            variables.write("gabor frequency = " + str(gabor_freq) + "\n")
            variables.write("gabor phase = " + str(gabor_phase) + "\n")
            variables.write("gabor sigma = " + str(gabor_sigma) + "\n")
            variables.write("gabor radius = " + str(gabor_radius) + "\n")
            variables.write("maxima detection radius = " + str(r_maxima) + "\n")
            variables.write("number of surrounding pixels taken into account per cell: " + str(reach) + "\n")
            variables.write("start time (ymd_h) = " + str(timestamp) + "\n")
            variables.write("time for analysis = " + str(int((time.time() - start_time) / 60)) + "min / " + str(
                round(((time.time() - start_time) / 3600), 2)) + "h" + "\n")
            variables.write("number of cells = " + str(n_cells))

    # close all h5 files
    closeall()
    print("script runtime: " + str(int((time.time() - start_time) / 60)) + "min / " + str(
        round(((time.time() - start_time) / 3600), 2)) + "h")
    print("Continuing with next file")

# shutdown
if shutdown == True:
    os.system('shutdown -s -t 300')  # to abort accidental shutdown on windows: "shutdown /a"
# In[code graveyard]
