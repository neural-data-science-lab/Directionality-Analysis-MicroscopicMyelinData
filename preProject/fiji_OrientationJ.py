from ij import IJ, Prefs
from ij import WindowManager, ImagePlus
from ij.measure import ResultsTable
import math

side = 'Left'
patch_size = int(math.floor(92.25))
filename = 'test_C03_smooth3D_bg95_frangi.tif'
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/test/'
outputpath = path+"test_pyFiji/"
name = side+str(patch_size)
img = IJ.openImage(path + filename)

width = img.getDimensions()[0]
print(width)
height = img.getDimensions()[1]
x_bound = width/patch_size
y_bound = height/patch_size

#macrotest = """
#run("OrientationJ Distribution", "tensor=2.0 gradient=0 radian=on table=on min-coherency=0.0 min-energy=0.0 ");
#"""
#IJ.runMacro(macrotest)

#create patches and run OrientationJ on them
for i in range(2):
	for j in range(2):
		img2 = img.duplicate()
		k = i*patch_size
		l = j*patch_size
		img2.setRoi(k, l, patch_size, patch_size)
		IJ.run(img2, "Crop", "")
		IJ.run(img2,"OrientationJ Distribution","tensor=2.0 gradient=0 radian=on table=on min-coherency=0.0 min-energy=0.0")
		ResultsTable.getResultsTable().save(outputpath + name + str(i)+"_"+str(j)+"OriJ.csv")
		IJ.selectWindow("OJ-Distribution-1")
		IJ.run("Close")
		IJ.selectWindow("OJ-Histogram-1-slice-1")
		IJ.run("Close")
		IJ.selectWindow("OJ-Histogram-1-slice-1")
		IJ.run("Close")
		

