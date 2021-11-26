from ij import IJ, Prefs
from ij import WindowManager, ImagePlus
from fiji.analyze.directionality import Directionality_

side = 'Left'
patch_size = int(round(92.25))
filename = 'LeftPR012_l_ACx_C03_bg.tif'
path = '/ptmp/muellerg/PR012_l_ACx/' #'/home/muellerg/nas/Gesine_Muellerg/new_substacks_gesine/PR012_l_ACx/'
name = 'PR012_l_ACx'

name_save = side+name+'_'+str(patch_size)+'_Fiji_Directionality_'
img = IJ.openImage(path + filename)

width = img.getDimensions()[0]
height = img.getDimensions()[1]
x_bound = width/patch_size
y_bound = height/patch_size
print('x0')

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
		print('x1')
		window = name_save+str(i)+"_"+str(j)+".csv"
		IJ.selectWindow(window)
		IJ.run("Close")
		
		
