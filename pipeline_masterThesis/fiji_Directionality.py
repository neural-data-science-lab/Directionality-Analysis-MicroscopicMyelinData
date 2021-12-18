from ij import IJ, Prefs
from ij import WindowManager, ImagePlus
from fiji.analyze.directionality import Directionality_

#patch_size = int(round(92.25))
#filename = 'PR012_l_ACx_C03_frangi.tif'
#path = '/ptmp/muellerg/PR012_l_ACx/'
#name = 'PR012_l_ACx'

name_save = name+'_'+str(patch_size)+'_Fiji_Directionality_'
img = IJ.openImage(path + filename)
img.setRoi(x_start, y_start, x_end, y_end)
IJ.run(img, "Crop", "")

width = img.getDimensions()[0]
height = img.getDimensions()[1]
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
		IJ.saveAs("Results", path+name_save+str(i)+"_"+str(j)+".csv")
		window = name_save+str(i)+"_"+str(j)+".csv"
		IJ.selectWindow(window)
		IJ.run("Close")
		
		
