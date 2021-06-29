import imagej
ij = imagej.init(headless=False) #'C:/Users/Gesine/Downloads/Fiji.app'; 'net.imagej:imagej+net.imagej:imagej-legacy'

from jnius import autoclass
autoclass('java.lang.System').out.println('Hello world')

from matplotlib import pyplot as plt
from skimage import io
import numpy as np


filename = "test_C03_smooth3D_bg95_sato.tif"
path = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/"
outputpath = path + "test_C03_smooth3D_bg95_sato_dice80_py/"
name = "py_dice80_"
img = io.imread(path + filename)
#ij.py.show(img[0])
#ij.ui().show('sato', ij.py.to_java(img)) #visualization such as in fiji itself

## create plugin
plugin = 'Directionality'
args = {
    'Method': 'Local gradient orientation',
    'Nbins': 90,
    'Histogram start': -90,
    'Histogram end': 90,
    'Build orientation map': 1,
    'Display table': 1,
}

width = img.shape[2]
height = img.shape[1]
depth = img.shape[0]

test_img = img[:,0:80, 0:80]
t = ij.ui().show('sato', ij.py.to_java(test_img))

patch_size = 80
x_bound = width/patch_size
y_bound = height/patch_size
for i in range(x_bound):
    for j in range(y_bound):
        img_patch = img[:, i*patch_size:i*patch_size+patch_size, j*patch_size:j*patch_size+patch_size]
        t = ij.ui().show('patch3D', ij.py.to_java(img_patch))
        ij.py.run_plugin(plugin, args)

'''issues
- Window Manager: windowManager = jnius.autoclass('ij.WindowManager') does not work
- no idea how to save table of directionality analysis'''