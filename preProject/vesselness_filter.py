import os
import timeit
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.filters import frangi, meijering, sato
from skimage.io import imsave

def normalize(image):
    min_val=np.min(image)
    max_val=np.max(image)
    return (image-min_val)/(max_val-min_val)

path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/'
name_data = 'ProbabilityMaps.tif'
path_data = os.path.join(path, name_data)
data = io.imread(path_data)

start = timeit.default_timer()
d = {}
for z in range(data.shape[0]):
    d[str(z)] = sato(data[z], black_ridges = False)
f = np.array([normalize(d[v]) for v in d.keys()])*255
imsave(path+"test_C03_weka_sato.tif", f.astype('uint16'))
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(round(execution_time,2))+" seconds")

# Plot results for one slice
slice = 0
fig, ax = plt.subplots(figsize=(10, 10), dpi=180, ncols=2, nrows=2)
ax[0,0].imshow(data[slice], cmap="gray")
ax[0,0].set_title('Image: smooth, background substraction')
ax[0,1].imshow(frangi(data[slice], black_ridges = False), cmap="gray")
ax[0,1].set_title('Frangi filter result')
ax[1,0].imshow(meijering(data[slice], black_ridges = False), cmap="gray")
ax[1,0].set_title('Meijering filter result')
ax[1,1].imshow(sato(data[slice], black_ridges = False), cmap="gray")
ax[1,1].set_title('Sato filter result')
fig.tight_layout()
plt.show()
