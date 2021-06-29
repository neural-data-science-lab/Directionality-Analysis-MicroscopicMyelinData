def chunk_generator(img_shape,chunk_size,overlap):
    '''
    Returns a sequence of coordinates every time it is called with next() that can be used to cycle through 3D arrays in blocks.

    Inputs:
    img_shape: image shape(z,y,x)
    chunk_size: desired chunk size (z,y,x)
    overlap: overlap (in pixels) on every side of the chunk

    Outputs:
    6 integers giving the start & end coordinates in all axes in the following order:
    xstart, xend, ystart, yend, zstart, zend

    to do:
        rest of image calculation, uneven boundaries
        n-dimensional image compatibility
    '''

    z_start = 0 
    z_end   = chunk_size[0]
    y_start = 0
    y_end   = chunk_size[1]
    x_start = 0
    x_end   = chunk_size[2]
    
    while x_end <= img_shape[2]: #if x_end exceeds x boundary of image, all is done

        yield (z_start,z_end,y_start,y_end,x_start,x_end)

        z_start = z_start + chunk_size[0]-2*overlap
        z_end   = z_start + chunk_size[0]
        
        # if z_end exceeds img shape: move y_start (and reset z_start)
        if z_end > img_shape[0]:
            y_start = y_start + chunk_size[1]-2*overlap
            y_end   = y_start + chunk_size[1]
            z_start = 0
            z_end   = chunk_size[0]
        
        # if z_end AND y_end exceed img shape: move x_start (and reset y_start and z_start)
        if y_end > img_shape[1]:
            x_start = x_start + chunk_size[2]-2*overlap
            x_end   = x_start + chunk_size[2]
            z_start = 0
            z_end   = chunk_size[0]
            y_start = 0
            y_end   = chunk_size[1]

    yield z_start,z_end,y_start,y_end,x_start,x_end

###### does not work??
import os
import skimage.io as io
import numpy as np

name_data = 'RightZ50_smooth2_bg95_sato.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Datensatz-0705/'
path_data = os.path.join(path, name_data)
data = io.imread(path_data)
cs = [0,80,80]
ImageShape = [data.shape[0], data.shape[1], data.shape[2]]
chunks = chunk_generator(img_shape = ImageShape, chunk_size = cs, overlap = 0)
next(chunks)
