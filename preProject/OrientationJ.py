import numpy as np
from skimage import feature
import os
import skimage.io as io
import pandas as pd
import matplotlib.pyplot as plt
import math

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def OrientationJ(img, sigma, direction):
    """OrientationsJ's dominant direction and orientation """
    dom_ori = []
    hist_ori = []
    hist_ori.append(direction)
    dim = img.shape[0]
    for i in range(dim):
        axx, axy, ayy = feature.structure_tensor(
            img[i].astype(np.float32), sigma=sigma, mode="reflect", order="rc"
        )
        dom_ori.append(np.rad2deg(np.arctan2(2 * axy.mean(), (ayy.mean() - axx.mean())) / 2))
        o = np.rad2deg(np.arctan2(2 * axy, (ayy - axx)) / 2)
        d = np.digitize(o, direction, right=True)
        h = np.histogram(d, bins=90, range=(0, 89))[0]
        #hist_ori.append(pd.DataFrame(np.vstack([direction, h / np.sum(h)]).T))
        hist_ori.append(h / np.sum(h))
    dom_ori = np.stack(dom_ori, axis=0)
    ori = np.stack(hist_ori, axis=1)

    """OrientationJ's output for
    * coherence
    * energy
    eps = 1e-20
    l1, l2 = feature.structure_tensor_eigenvalues([axx, axy, ayy])
    coh = ((l2 - l1) / (l2 + l1 + eps)) ** 2
    ene = np.sqrt(axx + ayy)
    ene /= ene.max()"""

    return dom_ori, ori


###### main ######
side = 'Left'
patch_size = int(math.floor(92.25))
filename = 'test_C03_smooth3D_bg95_frangi.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/test/'
outputpath = path+"dir_"+str(patch_size) +"/"
name = side+str(patch_size)
data = io.imread(path + filename)
#### initialize angles as in fiji directionality plugin
file = path +'dir_92/Left92' + str(0) + '_' + str(0) + '.csv'
path_patch0 = os.path.join(path, file)
patch0 = pd.read_csv(path_patch0, encoding = "ISO-8859-1")
patch0.rename(columns={'Direction (°)': 'Direction'}, inplace=True)
direction = patch0['Direction']
titles = patch0.keys()[1:][::2]

width = data.shape[2]
height = data.shape[1]
x_bound = int(width/patch_size)
y_bound = int(height/patch_size)
sigma = 2
for i in range(x_bound):
    d = []
    for j in range(y_bound):
        img = data[:, j*patch_size:j*patch_size+patch_size, i*patch_size:i*patch_size+patch_size]
        dom_dir, ori = OrientationJ(img, sigma, direction)
        ori = pd.DataFrame(ori)
        ori.to_csv(outputpath+name+'Ori'+str(i)+'_'+str(j)+'.csv', index=False)
        d.append(dom_dir)
    d = pd.DataFrame(d)
    d.to_csv(outputpath + name + 'domOri' + str(i) + '.csv', index=False)







'''code graveyard
### neglect 0.0 -> digitize -> histogram ???
ori_nonzero = ori[np.nonzero(ori)]
plt.imshow(img,cmap='gray')
plt.imshow(ori,cmap='gray')'''

