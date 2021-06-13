"""
Microscopy toolbox
"""
import numpy as np
import scipy.ndimage as ndi
import math as m
import random
import skimage.morphology as morph
import copy
import numpy
import matplotlib.pyplot as plt
import skimage.morphology
import skimage.io
import scipy.ndimage
from scipy import optimize
import functools
from matplotlib.widgets import Slider
import h5py
import os
import tifffile as tf
import scipy.spatial
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


class Image():
    '''
    Class for working with 3d volumes of microscopy data.
    '''

    def __init__(self, data):
        if isinstance(data, str):
            if data == 'test':
                data = artificial_data(cell_radius_range=[6, 10, 14, 18, 22])
            else:
                pass  # treat as filename and load data
        self.data = data

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    def __add__(self, other):
        new = copy.deepcopy(self)
        if isinstance(other, type(self)):
            new.data = self.data + other.data
        else:
            new.data = self.data + other
        return new

    __radd__ = __add__

    def plot(self, z=None):
        if not z:
            z = self.data.shape[0] // 2
        fig, ax = plt.subplots()
        ax.matshow(self.data[int(z), :, :], interpolation='none', cmap='gray')
        fig.show()

    def zeros(self):
        return Image(numpy.zeros_like(self.data))

    def zscore(self):
        self.data = self.data - numpy.mean(self.data)
        # numpy.subtract(self, numpy.mean(self),out=self, casting="unsafe")
        self.data = self.data / numpy.std(self.data)

    def threshold(self, thresh=None):
        if not thresh:
            thresh = self.data.min() + self.data.ptp() / 3
        self.data[self.data < thresh] = 0

    def blur(self, sigma=6, inplace=False):
        if inplace:
            self.data = scipy.ndimage.filters.gaussian_filter(self.data, sigma=sigma)
        else:
            return Image(scipy.ndimage.filters.gaussian_filter(self.data, sigma=sigma))

    def pad(self, n):
        n = int(n)
        if numpy.sign(n) == 1:
            self.data = numpy.pad(array=self.data, pad_width=n, mode='edge')  # pad
        else:  # negative n
            n *= -1
            self.data = self.data[n:-n, n:-n, n:-n]  # crop


def describe(img, big=False):
    """
    describes properties of a given 3D numpy image
    img: input array
    big: if True, only plots the middle Z slice of the image
    """
    print("shape:" + str(img.shape))
    print("max:" + str(np.max(img)))
    print("min:" + str(np.min(img)))
    print("dtype:" + str(img.dtype))
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("intensity")
    ax.set_ylabel("n of pixels")
    if big == True:
        plt.hist(img[img.shape[0] // 2].ravel(), bins=256, range=(np.min(img), np.max(img)), fc='k', ec='k')
    else:
        plt.hist(img.ravel(), bins=256, range=(np.min(img), np.max(img)), fc='k', ec='k')
    plt.show()


def artificial_data(edge_length=500, n_cells=50, cell_radius_range=[5, 7, 10, 15], cell_intensity=100,
                    background_intensity=50, noise_meter=0):
    '''
    Generates artificial 3D microscopy data with bright blobs inside. They can overlap.

    edge_length: edge length of the image cube
    n_cells: number of cells in the dataset
    cell_radius_range: list of radii, one of is randomly chosen for each cell
    noise_meter: gaussian noise to added to the image. 0=no noise
    '''
    img = np.zeros((edge_length, edge_length, edge_length), dtype="uint16")

    pad_length = cell_radius_range[-1] + 2
    img = np.pad(array=img,
                 pad_width=pad_length,
                 mode="constant",
                 constant_values=0)

    img[img < 1] = background_intensity

    for n_cells in range(n_cells):
        r_cell = cell_radius_range[random.randint(0, len(cell_radius_range) - 1)]
        cell_body = morph.ball(r_cell)

        randx = random.randint(r_cell, edge_length + r_cell)
        randy = random.randint(r_cell, edge_length + r_cell)
        randz = random.randint(r_cell, edge_length + r_cell)

        cell_body_array = np.zeros_like(img)
        cell_body_array[randx - r_cell:randx + r_cell + 1,
        randy - r_cell:randy + r_cell + 1,
        randz - r_cell:randz + r_cell + 1] = cell_body

        # adding single cell to whole image
        img = img + cell_body_array
        print("inserted cell #" + str(n_cells))

    # cropping image down to original size
    img = img[cell_radius_range[-1]:-cell_radius_range[-1],
          cell_radius_range[-1]:-cell_radius_range[-1],
          cell_radius_range[-1]:-cell_radius_range[-1]]

    # set intersections of cells to respective intensity
    img[img > background_intensity] = cell_intensity

    # add noise
    if noise_meter > 0:
        noise = np.random.normal(loc=0, scale=noise_meter, size=img.shape)
        img = img + noise

    img = img[pad_length:-pad_length, pad_length:-pad_length, pad_length:-pad_length]

    return img

    if plot:
        fig, ax = plt.subplots()
        ax.matshow(img[z, :, :], interpolation='none', cmap='gray')
        fig.colorbar(img[z, :, :], ax=ax)
        fig.show()
    if file_name:
        skimage.io.imsave(file_name, img.astype("float32"))
    return img, centers


def _gabor_shell(edge_length=50, radius=6, sigma=None, freq=.1, phase=4.7124, z_scale_factor=1):
    '''
    Internal function. Don't call directly.
    '''
    if not sigma:
        sigma = 4.  # if radius > 6 else radius
    if edge_length % 2 == 0:
        edge_length = edge_length + 1
    size_vector = numpy.arange(0, edge_length, 1) - edge_length / 2
    z, y, x = numpy.meshgrid(size_vector, size_vector, size_vector)
    y *= z_scale_factor
    A = (-2 * numpy.pi * sigma ** 2)
    r = numpy.sqrt(z ** 2 + y ** 2 + x ** 2)
    kernel = (1 / A) * numpy.exp(-1 * numpy.pi * (r - radius) ** 2 / sigma ** 2) * numpy.cos(
        2 * numpy.pi * freq * (r - radius) + phase)
    kernel = kernel / numpy.sum(kernel ** 2)
    return kernel


def gabor_shell(edge_length=50, radius=6, sigma=None, freq=.1, z_scale_factor=1, plot=False):
    phase = _compute_phase(edge_length=edge_length, radius=radius, sigma=sigma, freq=freq,
                           z_scale_factor=z_scale_factor)
    kernel = _gabor_shell(edge_length=edge_length, radius=radius, sigma=sigma, freq=freq, phase=phase,
                          z_scale_factor=z_scale_factor)
    if plot:
        fig, ax = plt.subplots()
        ax.matshow(kernel[int(edge_length / 2), :, :], interpolation='none', cmap='gray')
        fig.show()
    return kernel


def _compute_phase(edge_length=50, radius=6, sigma=None, freq=.1, z_scale_factor=1, plot=False):
    '''
    Internal function. Don't call directly.
    '''
    gabor = functools.partial(_gabor_shell, edge_length=edge_length, radius=radius, sigma=sigma, freq=freq,
                              z_scale_factor=z_scale_factor)
    f = lambda p: gabor(phase=p).sum()
    sol = optimize.root_scalar(f, bracket=[3, 4.8], method='brentq')
    if not sol.converged:
        raise ValueError('Phase optimization did not converge!')
    return sol.root


def interactive_gabor_shell(edge_length=50, radius_0=6, sigma_0=8, freq_0=0.1, phase_0=4.7124):
    '''
    Displays an interactive gabor shell - for demonstration purposes only
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Adjust the subplots region to leave some space for the sliders and buttons
    fig.subplots_adjust(bottom=0.3)

    # Draw the initial plot
    # The 'line' variable is used for modifying the line later
    shell = _gabor_shell(edge_length=edge_length, radius=radius_0, sigma=sigma_0, freq=freq_0, phase=phase_0)
    img = ax.imshow(shell[shell.shape[0] // 2, :, :], cmap='gray')

    # Add sliders for tweaking the parameters
    radius_slider_ax = fig.add_axes([0.25, 0.25, 0.65, 0.03])
    radius_slider = Slider(radius_slider_ax, 'Radius', 4, 20, valinit=radius_0)
    freq_slider_ax = fig.add_axes([0.25, 0.20, 0.65, 0.03])
    freq_slider = Slider(freq_slider_ax, 'Freq', 0.01, 1, valinit=freq_0)
    sigma_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    sigma_slider = Slider(sigma_slider_ax, 'Sigma', 1, 20, valinit=sigma_0)
    phase_slider_ax = fig.add_axes([0.25, 0.10, 0.65, 0.03])
    phase_slider = Slider(phase_slider_ax, 'Phase', 0, 6.28, valinit=phase_0)

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):
        data = _gabor_shell(edge_length=edge_length, radius=radius_slider.val, sigma=sigma_slider.val,
                            freq=freq_slider.val, phase=phase_slider.val)[edge_length // 2, :, :]
        data = data / data.max()
        img.set_data(data)
        fig.canvas.draw_idle()

    radius_slider.on_changed(sliders_on_changed)
    freq_slider.on_changed(sliders_on_changed)
    sigma_slider.on_changed(sliders_on_changed)
    phase_slider.on_changed(sliders_on_changed)

    plt.show()


def rotate_dataset(cellcoords, n_cell, surfacecoords, ):  # unfinished, do not use
    '''
    UNFINISHED
    Rotate a dataset with the shape (n,3). Rotation is done so the new Z axis is orthogonal to the cortex surface.
    cellcoords: int (not uint) input array with shape (n,3) with n Being the number of cells and the second axis being xyz coordinates (in that order).
    n_cell: position of the "active" cell, which will the origin of the rotation.
    surfacecoords: array of coordinates of the cortex surface, in the same space as cellcoords. nearest coord will be used for rotation
    '''

    a = cellcoords[0]  # active cell
    b = cellcoords[1]  # nearest cortex surface
    v_ab = b - a  # vector from a to b

    # set active cell to origin (0,0,0)
    cellcoords = cellcoords - a

    ###find euler angles by reducing one dimension at each time, calculating the respective angle to turn around the axis at a time
    # gamm: angle between xy subvector and Y Axis -- rotation around Z axis
    v_y = np.array([0, 1])  # 2d y vector [x,y]
    v_ab_xy = v_ab[0:2]  # reduce 3d vector to XY plane
    # calculate angle between v_ab_xy and y vector
    gamm = m.acos((np.dot(v_ab_xy, v_y)) / (
                m.sqrt(v_ab_xy[0] * v_ab_xy[0] + v_ab_xy[1] * v_ab_xy[1]) * m.sqrt(v_y[0] * v_y[0] + v_y[1] * v_y[1])))
    gamm = m.degrees(gamm)  # angle to turn around Z axis

    # bet: angle between yz subvector and Z Axis -- rotation around X axis
    v_z = np.array([0, 1])  # 2d z vector [y,z]
    # v_z=v_z[2:4]
    v_ab_yz = v_ab[1:3]  # reduce vector to YZ plane
    # calculate angle between v_ab_yz and Z vector
    alph = m.acos((np.dot(v_ab_yz, v_z)) / (
                m.sqrt(v_ab_yz[0] * v_ab_yz[0] + v_ab_yz[1] * v_ab_yz[1]) * m.sqrt(v_z[0] * v_z[0] + v_z[1] * v_z[1])))
    alph = m.degrees(alph)  # angle to turn around X axis

    # bet: angle between xz subvector and X Axis -- rotation around Y axis
    # no need !

    # turn the dataset accordingly
    rot = R.from_euler("zyx", (gamm, 0, alph), degrees=True)  # initialize rotation
    coord_rotated = rot.apply(cellcoords)  # rotate

    if plot == True:  # plot 3D scatterplot
        labels = ("cell", "surface")
        labels_r = ("cell_r", "surface_r")
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(cellcoords[:, 0], cellcoords[:, 1], cellcoords[:, 2])  # before rotation, blue
        ax.scatter(coord_rotated[:, 0], coord_rotated[:, 1], coord_rotated[:, 2])  # after rotation, orange
        for i in range(len(labels)):
            ax.text(cellcoords[i, 0], cellcoords[i, 1], cellcoords[i, 2], '%s' % (labels[i]), size=10, zorder=1, )
            ax.text(coord_rotated[i, 0], coord_rotated[i, 1], coord_rotated[i, 2], '%s' % (labels_r[i]), size=10,
                    zorder=1, )
        ax.set_zlabel("Z")
        ax.set_ylabel("Y")
        plt.show()
        return coord_rotated


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


def gaborkernel(edge_length, sigma, freq, phase, radius, z_scale_factor):
    '''

    *** OLD VERSION ***

    Returns two gabor kernels (real and imaginary) for spheroid detection. Can be convolved with an image (using )
    
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


def chunk_generator(img_shape, chunk_size, overlap):
    '''
    Returns a sequence of coordinates every time it is called with next() that can be used to cycle through 3D arrays in blocks.

    Inputs:
    img_shape: image shape(z,y,x)
    chunk_size: desired chunk size (z,y,x)
    overlap: overlap (in pixels) on every side of the chunk

    Outputs:
    6 integers giving the start & end coordinates in all axes
    xstart, xend, ystart, yend, zstart, zend

    to do:
        rest of image calculation, uneven boundaries
        n-dimensional image compatibility
    '''

    z_start = 0
    z_end = chunk_size[0]
    y_start = 0
    y_end = chunk_size[1]
    x_start = 0
    x_end = chunk_size[2]

    while x_end <= img_shape[2]:  # if x_end exceeds x boundary of image, all is done

        yield (z_start, z_end, y_start, y_end, x_start, x_end)

        z_start = z_start + chunk_size[0] - 2 * overlap
        z_end = z_start + chunk_size[0]

        # if z_end exceeds img shape: move y_start (and reset z_start)
        if z_end > img_shape[0]:
            y_start = y_start + chunk_size[1] - 2 * overlap
            y_end = y_start + chunk_size[1]
            z_start = 0
            z_end = chunk_size[0]

        # if z_end AND y_end exceed img shape: move x_start (and reset y_start and z_start)
        if y_end > img_shape[1]:
            x_start = x_start + chunk_size[2] - 2 * overlap
            x_end = x_start + chunk_size[2]
            z_start = 0
            z_end = chunk_size[0]
            y_start = 0
            y_end = chunk_size[1]

    yield z_start, z_end, y_start, y_end, x_start, x_end


def edge_detection(img):
    '''
    Performs a sobel edge detection in x,y,z directions.(ndi.sobel)
    '''
    # Compute sobel filter along all three axes
    edges1 = ndi.sobel(img, axis=0)
    edges2 = ndi.sobel(img, axis=1)
    edges3 = ndi.sobel(img, axis=2)

    # Average images and z score image
    edges_sum = (edges1 + edges2 + edges3) / 3
    edges_sum[edges_sum < 0] = edges_sum[edges_sum < 0] * (-1)

    # gauss img
    edges_sum = ndi.gaussian_filter(edges_sum, sigma, )
    return edges_sum


def plot(self, z=None, overlay=None):
    '''
    Opens an interactive volume plot. Scroll through the first dimension with
    the a and d keys. skimage.external.tifffile.imshow has similar functionality.
    if overlay is an ndarray or Volume of the same dimensions, it is overlayed
    transparently.
    '''

    def scroll(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'a':
            ax.index = (ax.index - 1) % ax.volume.shape[0]
        elif event.key == 'd':
            ax.index = (ax.index + 1) % ax.volume.shape[0]
        ax.images[0].set_array(ax.volume[ax.index, :, :])
        if hasattr(ax, 'volume2'):
            ax.images[1].set_array(ax.volume2[ax.index, :, :])
        ax.set_title(f'z = {str(ax.index)}')
        fig.canvas.draw()

    fig, ax = plt.subplots()
    ax.volume = self.data
    if isinstance(overlay, Volume) and self.data.shape == overlay.data.shape:
        ax.volume2 = numpy.ma.masked_less(overlay.data, 0.5)
    elif isinstance(overlay, numpy.ndarray) and self.data.shape == overlay.shape:
        ax.volume2 = numpy.ma.masked_less(overlay, 0.5)
    if not z:
        z = self.data.shape[0] // 2
    ax.index = z
    ax.imshow(ax.volume[ax.index, :, :], cmap='gray')
    if hasattr(ax, 'volume2'):
        ax.imshow(ax.volume2[ax.index, :, :], alpha=0.5)
    fig.canvas.mpl_connect('key_press_event', scroll)
    plt.show()


def count_digits(n):  # counts the digits of a given number
    count = 0
    while (n > 0):
        count = count + 1
        n = n // 10
    return count


def preprocessing_terastitcher(input_path, output_path, raw_filename, n_tiles_x=8, n_tiles_y=3, n_tiles_z=801,
                               channels=(r"C00", r"C01"), input_format=r".tif"):
    """
    Concatenates LaVision Ultramicroscope II raw data in Z in multiple channels and in mosaics.
    Outputs correct file and folder structure to continue with Terastitcher Stitching
    So far only works for Data generated with 20x LaVision UltraII objective! 
    
    so far, only works for images in range smaller than 10000µm 
    input strings as r"" (raw) strings

        n_channels: number of imaged channels
    
    script options:
        input_path: folder path to where raw images are saved
        output_path: folder path to where preprocessed images are saved
        raw_filename: filename up to the point where Z stack and mosaic coordinates are 
            (e.g. r"\11-11-59_AJ003_HuCD561_Topro640_UltraII")
        channels: list of channels, e.g. (r"C00", r"C01")

    dimensions: XYCZT
    """
    # microscope settings
    overlap = 0.10
    x_tile_microns = 1170
    y_tile_microns = 1387
    final_shape = (n_tiles_z, 2560, 2160, len(channels))  # final shape of a single concatenated tile (z,y,x,c)

    # initialize variables, fix index so it starts at 0
    output_name = r"\stack"
    n_tiles_x = n_tiles_x - 1
    n_tiles_y = n_tiles_y - 1
    current_x = 0
    current_y = 0
    n_iter = 0
    x_tile_offset = int(x_tile_microns - (int(x_tile_microns * overlap)) * 2)  # calculate offset in µm (minus overlap)
    y_tile_offset = int(y_tile_microns - (int(y_tile_microns * overlap)) * 2)

    while current_x <= n_tiles_x:  # iterate through XY tiles
        # while n_iter < 1:

        n_iter += 1

        # multiply by 10 to get to tenths of µm format needed for terastitcher
        x_dist = str(current_x * x_tile_offset * 10)
        y_dist = str(current_y * y_tile_offset * 10)

        # get x_dist and y_dist to 6 digits
        while len(x_dist) < 6:
            x_dist = "0" + x_dist

        while len(y_dist) < 6:
            y_dist = "0" + y_dist

        current_path = output_path + "\\" + str(x_dist)
        if os.path.isdir(current_path) == False:
            os.mkdir(current_path)

        current_path += "\\" + str(x_dist) + "_" + str(y_dist)
        if os.path.isdir(current_path) == False:
            os.mkdir(current_path)

        stack = np.zeros(final_shape, dtype=np.uint16)  # initialize empty image

        print("concatenating channels for mosaic tile Y=" + str(current_y) + " X=" + str(
            current_x))  # concatenate all channels for given tile

        # define current tile in file handle
        prefix = raw_filename + r"[0" + str(current_y) + r" x 0" + str(
            current_x) + r"]_"  # prefix of filename prefix (before z index)
        suffix = input_format  # suffix of filename (after z index)

        for channel in range(len(channels)):  # iterate through channels

            prefix_channel = prefix + channels[channel] + r"_xyz-Table Z"  # adding channel to filename prefix

            print("starting concatenating for " + channels[channel])

            for z in range(final_shape[0]):  # iterate through z

                z_number = (4 - len(str(z))) * "0" + str(z)  # generate four-digit Z number

                if z % 100 == 0:  # timekeeping
                    print("active Channel: " + channels[channel] + " -- active z slice: " + str(z))

                current_z = tf.imread(os.path.join(input_path, prefix_channel) + z_number + suffix,
                                      multifile=False)  # adding Z index to filename and opening the relevant file
                stack[z, :, :, channel] = current_z

            print("Done with " + channels[channel] + "!")

        stack = stack.astype(r"uint16")
        tf.imsave(file=current_path + r"\000000.tif", data=stack, bigtiff=True, dtype="uint16",
                  planarconfig="CONTIG")  # save as tiff

        current_y += 1

        if current_y > n_tiles_y:
            current_y = 0
            current_x += 1
            continue

        else:
            print("Saving File")


def convert_to_hdf5(input_path, output_path, dataset_name=u'/t0/channel0', compression=0, chunks=True):
    '''
    Converts a 3D .tiff to .h5
    input_path: direct path to .tif
    output_path: direct path to .h5 (must not be already present)
    '''
    img = tf.imread(input_path)
    results = h5py.File(output_path, "w")
    results.create_dataset(dataset_name,
                           dtype=img.dtype,
                           chunks=chunks,
                           compression=compression,
                           shape=(img.shape))
    results[dataset_name][:] = img
    results.close()

# to put attributes to a dataset:
# file[u'dataset1'].attrs['attr1']=np.string_('hi there')


###
### code graveyard
###

# def concatenate_z(input_path,output_path,n_channels,):
#     """
#     WORK IN PROGRESS

#     takes raw LaVision UltraII output and concetanates the files into large z-stacks
#     input_path = r"C:\Data\20190716_AJ003_Overview\190716_AJ003_HuCD568_Topro640_12-39-56" #path to raw files
#     output_path = r"C:\Data\20190716_AJ003_Overview\190716_AJ003_HuCD568_Topro640_12-39-56\results" #output path for concatenated h5 file
#     stack_filename = r"\AJ003_overview" #filename of resulting stack
#     prefix_singlestack =r"\12-39-56_AJ003_HuCD568_Topro640_UltraII_"
#     final_shape = (1047,2560,2160) #final_shape of finished dataset (z,y,x)
#     n_channels = number of channels
#     """

#     n_z_digits = count_digits(final_shape[0]) #number of digits of Z slices
#     channels = (r"C00", r"C01", r"C02", r"C03", r"C04") #list of possible channels (as given in the filenames)
#     channels = channels[0:n_channels] #list of channels

#     stack = h5py.File(output_path + stack_filename + r".h5", "w") #initialize hdf5 file

#     for channel in channels:

#         print ("starting concetanating for " + channel)

#         prefix = prefix_singlestack + channel + r"_xyz-Table Z" #prefix of filename (before z index)
#         suffix = r".ome.tif" #suffix of filename (after z index)

#         stack.create_dataset(u"stack_" + str(channel),
#                             dtype="uint16",
#                             chunks=True,
#                             compression=None, 
#                             shape=(final_shape))

#         for z in range(final_shape[0]):

#             z_number = (n_z_digits-1)*"0" + str(z) #for single digit z

#             if z > 9:
#                 z_number = (n_z_digits-2)*"0" + str(z) #for double digit z

#             if z > 99:
#                 z_number = (n_z_digits-3)*"0" + str(z) # etc

#             if z > 999:
#                 z_number = (n_z_digits-4)*"0" + str(z)

#             if z % 20 == 0:
#                 print ("active Channel: " + channel + " -- active z slice: " + str(z))

#             current_z = tf.imread(input_path + prefix + z_number + suffix, multifile = False)
#             stack[u"stack_" + str(channel)][z,:,:] = current_z

#         print ("Done!")

#         stack.close()

#     print("done")


# def test_volume(radii=None, file_name=None, plot=False):
#     '''
#     img = test_volume(radii=[6, 10, 14, 18, 22], plot=True)
#     add noise?
#     '''
#     background_intensity = 850 #background intensity
#     cell_intensity = 1000
#     n_cells = len(radii) #number of cells
#     n_rows = numpy.floor(numpy.sqrt(n_cells))
#     n_cols = numpy.ceil(n_cells / n_rows)
#     max_r = numpy.max(radii) # max radius
#     max_d = 2 * max_r # mas diameter
#     x_len = int(n_cols * max_d) # width of volume
#     y_len = int(n_rows * max_d) # hight of volume
#     z_len = int(max_d + 1) # depth of volume, always odd so that we have a middle plane
#     z = int(max_r + 1) # middle z plane idx
#     centers = []
#     img = numpy.full((z_len, y_len, x_len), background_intensity, dtype="uint16")
#     for idx, radius in enumerate(radii):
#         cell_body = skimage.morphology.ball(radius) * (cell_intensity-background_intensity) + background_intensity
#         x = idx % n_cols
#         y = idx // n_cols
#         x = int(max_d * x + max_r)
#         y = int(max_d * y + max_r)
#         img[z-radius-1:z+radius, y-radius-1:y+radius, x-radius-1:x+radius] = cell_body
#         centers.append((z,y,x))

#     # #add noise
#     # noise = np.random.normal(loc=0,scale=noisiness,size=img.shape)
#     # img_noisy = img + noise


# def gabor_correct_radius(edge_length=50, radius=6, sigma=None, freq=.1, z_scale_factor=1, plot=False):
#     '''
#     Internal function. Don't use.
#     '''
#     if radius > 13:
#         return gabor_shell(edge_length=edge_length, radius=radius, sigma=sigma, freq=freq, z_scale_factor=z_scale_factor, plot=plot)
#     else:
#         phase = _compute_phase(edge_length=edge_length, radius=radius, sigma=sigma, freq=freq, z_scale_factor=z_scale_factor)
#         # compute real r from phase here:
#         #f = lambda r: gabor(radius=r)
#         #sol = optimize.root_scalar(f, bracket=[3, 4.8], method='brentq')

# def check_phases():
#     '''
#     Internal function. Don't call directly.
#     '''
#     plt.figure()
#     for r in range(3, 21):
#         s = []
#         phases = numpy.linspace(0, numpy.pi*2, 40)
#         for p in phases:
#             gabor = _gabor_shell(edge_length=50, freq=0.1, phase=p, radius=r)
#             s.append(gabor.sum())
#         plt.plot(phases, s, label=str(r))
#     plt.legend()
#     plt.show()
