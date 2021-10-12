from ij import IJ, Prefs
from ij import WindowManager, ImagePlus
from fiji.analyze.directionality import Directionality_

side = 'Left'
patch_size = 80
filename = side+'50_smooth2_bg95_frangi.tif'
path = '/media/muellerg/Data SSD/Gesine/Data/'
outputpath = path+side+'_frangiTest_'+str(patch_size)+'/'
name = side+str(patch_size)
img = IJ.openImage(path + filename)

width = img.getDimensions()[0]
height = img.getDimensions()[1]
if ( (width % patch_size) != 0  ): print("Adjust patch size.")
if ( (height % patch_size) != 0  ): print("Adjust patch size.")
x_bound = width/patch_size
y_bound = height/patch_size

#create patches and run directionality on them
for i in range(x_bound):
	for j in range(y_bound):
		img2 = img.duplicate()
		k = i*patch_size
		l = j*patch_size
		img2.setRoi(k, l, patch_size, patch_size)
		IJ.run(img2, "Crop", "")
		dir = Directionality_()
		dir.setImagePlus(img2)
		dir.setMethod(Directionality_.AnalysisMethod.LOCAL_GRADIENT_ORIENTATION)
		dir.setBinNumber(90)
		dir.setBinStart(-90)
		dir.setBuildOrientationMapFlag(True)
		dir.computeHistograms()
		dir.fitHistograms()
		dir.displayResultsTable().show("Directionality histograms")
		IJ.saveAs("Results", outputpath+name+str(i)+"_"+str(j)+".csv")
		window = name+str(i)+"_"+str(j)+".csv"
		IJ.selectWindow(window)
		IJ.run("Close")
		
		