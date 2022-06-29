'''
author: Gesine Müller
    apply a squared function to approximate the cortex mask (for every z-file separately)
'''

import os
import numpy as np
from skimage import io
from scipy.optimize import curve_fit
from skimage.io import imsave
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('path', type=str)
args = parser.parse_args()

# define the objective function (squared function)
def objective(x, a, b, c):
    return a * x + b * x**2 + c

def fit_squaredFct_cortexMask(data):
    e = np.swapaxes(data, 0, 1)
    cortexMask = np.zeros(e.shape, dtype=np.uint8)
    samples = np.array((np.array(np.where(e == 255)).T[:, 0], np.array(np.where(e == 255)).T[:, 1])).T
    sample_sorted = np.array(sorted(samples, key=lambda x: x[1]))
    limit = np.where(sample_sorted[:, 1] == 2000)[0][0]
    popt, pcov = curve_fit(objective, samples[0:limit, 1], samples[0:limit, 0])
    xvalues = np.arange(0, cortexMask.shape[1], 1)
    fittedCurve = objective(xvalues, *popt)
    for column in range(cortexMask.shape[1]):
        for row in range(cortexMask.shape[0]):
            if row <= fittedCurve[column]:
                cortexMask[row, column] = 0
            else:
                cortexMask[row, column] = 255
    cortexMask = np.swapaxes(cortexMask, 0, 1)

    return cortexMask


#path  = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/'
#name = 'PR012_l_ACx'
filename = args.name + '_C00_cortex-edges.tif'
image = io.imread(os.path.join(args.path, filename))
cortexMask = []
batch_size = 5
depth = image.shape[0]

for batch in range(int(image.shape[0] / batch_size)):
    mask = []
    for z in range(batch_size):
        mask.append(fit_squaredFct_cortexMask(image[batch*batch_size+z]))
    cortexMask.append(mask)
cortexMask = np.vstack(cortexMask)
imsave(args.path + args.name + "_" + 'C00_cortex.tif', cortexMask.astype('uint8'))




'''code graveyard
import matplotlib.pyplot as plt
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/'
filename = 'PR012_l_ACx_C00_cortex.tif'
edges = 'PR012_l_ACx_C00_cortex-2D_edges.tif'
data = io.imread(os.path.join(path, filename))[0]
e = np.swapaxes(io.imread(os.path.join(path, edges)), 0, 1)
cortexMask = np.zeros(e.shape)

samples = np.array((np.array(np.where(e==255)).T[:,0], np.array(np.where(e==255)).T[:,1])).T
sample_sorted = np.array(sorted(samples, key=lambda x: x[1]))

limit = np.where(sample_sorted[:,1]==1700)[0][0]
popt, pcov = curve_fit(objective, samples[0:limit,1], samples[0:limit,0])

plt.imshow(e)
plt.scatter(samples[:,1], samples[:,0])
plt.scatter(samples[:,1], objective(samples[:,1], *popt))
xvalues = np.arange(0, cortexMask.shape[1],1)
plt.scatter(xvalues, objective(xvalues, *popt))
fittedCurve = objective(xvalues, *popt)
for column in range(cortexMask.shape[1]):
    for row in range(cortexMask.shape[0]):
        if row <= fittedCurve[column]:
            cortexMask[row,column] = 0
        else:
            cortexMask[row, column] = 255
plt.imshow(cortexMask)
'''