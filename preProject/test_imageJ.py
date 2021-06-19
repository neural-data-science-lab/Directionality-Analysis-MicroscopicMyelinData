import imagej
from skimage import io
import numpy as np

ij = imagej.init('C:/Users/Gesine/Downloads/Fiji.app', headless=False) #'C:/Users/Gesine/Downloads/Fiji.app'; r
ij.getVersion()

filename = "test_C03_smooth3D_bg95_sato.tif"
path = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/"
outputpath = path + "test_C03_smooth3D_bg95_sato_dice80_py/"
name = "py_dice80_"
img = io.imread(path + filename)

url = 'https://www.fi.edu/sites/fi.live.franklinds.webair.com/files/styles/featured_large/public/General_EduRes_Heart_BloodVessels_0.jpg'
image = io.imread(url)
image = np.mean(image, axis=2)
ij.py.show(image)