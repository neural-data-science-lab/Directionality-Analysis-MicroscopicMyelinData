import imagej
ij = imagej.init('C:/Users/Gesine/Downloads/Fiji.app', headless=False) #'C:/Users/Gesine/Downloads/Fiji.app'; 'net.imagej:imagej+net.imagej:imagej-legacy'

from skimage import io
import numpy as np
import scyjava
from scyjava import jimport
from jnius import autoclass
WindowManager = autoclass('ij.WindowManager')


name_data = 'test_C03_smooth3D_bg95_frangi.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/test/' #'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/'
outputpath = path + 'test_pyFiji/'
img = io.imread(path + name_data)
test_img = img[0,0:80,0:80]
ij.ui().show('frangi', ij.py.to_java(test_img))
#image = ij.py.from_java(img[0,0:80,0:80])
#ij.py.show(image, cmap='gray')

### both works somewhat the same ###
'''directionality_macro = """
run("Directionality", "method=[Local gradient orientation] nbins=90 histogram_start=-90 histogram_end=90 display_table");
"""
ij.py.run_macro(directionality_macro)'''

## create plugin
plugin = 'Directionality'
args = {
    'method': 'Local gradient orientation',
    'nbins': 90,
    'histogram_start': -90,
    'histogram_end': 90,
    'display_table': True
    }
close = """
selectWindow("Directionality histograms for frangi (using Local gradient orientation)")
run("Close")
run("Collect Garbage")
WindowManager.closeAllWindows()
"""
ij.py.run_plugin(plugin, args)
ij.py.run_macro(close)

### try OrientationJ ### nicht besser als in Fiji selbst -> zu viele fenster offen!!!
'''OrientationJ_macro="""
open('C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/testImage_artificial-fibers.tif')
run("OrientationJ Distribution", "tensor=2.0 gradient=0 radian=on table=on min-coherency=0.0 min-energy=0.0 ");
saveAs("Results", outputpath + "test_OrientationJ.csv")
"""
ij.py.run_marco(OrientationJ_macro)

plugin = 'OrientationJ Distribution'
args = {
    'tensor': 2.0,
    'gradient': 0,
    'radian': 0,
    'min-coherency': 0.0,
    'min-energy': 0.0,
    'table': 1,
    'histogram': 0
    }
ij.py.run_plugin(plugin, args)'''

WindowManager = jimport('ij.WindowManager')
current_image = WindowManager.getCurrentImage()

width = img.shape[2]
height = img.shape[1]
depth = img.shape[0]

patch_size = 80
x_bound = width/patch_size
y_bound = height/patch_size
for i in range(x_bound):
    for j in range(y_bound):
        img_patch = img[:, i*patch_size:i*patch_size+patch_size, j*patch_size:j*patch_size+patch_size]
        t = ij.ui().show('patch3D', ij.py.to_java(img_patch))
        ij.py.run_plugin(plugin, args)

