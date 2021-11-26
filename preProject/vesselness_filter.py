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

path = 'E:/MasterThesis/PR012_l-ACx/'
name_data = '11-53-16_PR012_UltraII[01 x 00]_C03_xyz-Table Z0335.ome.tif'
path_data = os.path.join(path, name_data)
data = io.imread(path_data)

start = timeit.default_timer()
vesselness = np.zeros((2, data.shape[1],  data.shape[2]))
for z in range(2):
    f = frangi(data[z], black_ridges = False)
    vesselness[z,:,:] = normalize(f)*65536
imsave(path+name_data+"_frangi.tif", vesselness.astype('uint16'))
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(round(execution_time,2))+" seconds")

''''# Plot results for one slice
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
'''