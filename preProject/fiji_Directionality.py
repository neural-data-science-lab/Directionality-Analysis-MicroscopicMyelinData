from ij import IJ, Prefs
from ij import WindowManager, ImagePlus
from fiji.analyze.directionality import Directionality_

filename = "test_C03_smooth3D_bg95.tif"
path = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/"
outputpath = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03_smooth3D_bg95/"
name = "directionalityLG_"
img = IJ.openImage(path + filename)

width = img.getDimensions()[0]
height = img.getDimensions()[1]
if ( (width % 60) != 0  ): print("Adjust patch size.")
if ( (height % 80) != 0  ): print("Adjust patch size.")
x_bound = width/60
y_bound = height/80

#create patches and run directionality on them
for i in range(x_bound):
	for j in range(y_bound):
		img = IJ.openImage(path + filename)
		k = i*60
		l = j*80
		img.setRoi(k, l, 60, 80)
		IJ.run(img, "Crop", "")
		dir = Directionality_()
		dir.setImagePlus(img)
		dir.setMethod(Directionality_.AnalysisMethod.LOCAL_GRADIENT_ORIENTATION)
		dir.setBinNumber(90)
		dir.setBinStart(-90)
		dir.setBuildOrientationMapFlag(True)
		dir.computeHistograms()
		dir.fitHistograms()
		#plot_frame = dir.plotResults()
		#plot_frame.setVisible(True)
		#data_frame = dir.displayFitAnalysis()
		#data_frame.setVisible(True) 
		#table = dir.displayResultsTable()
		#table.show("Directionality histograms")
		dir.displayResultsTable().show("Directionality histograms")
		IJ.saveAs("Results", outputpath+name+str(i)+"_"+str(j)+".csv")
		#stack = dir.getOrientationMap()
		#ImagePlus("Orientation map", stack).show()
		#Directionality_.generateColorWheel().show()
		window = name+str(i)+"_"+str(j)+".csv"
		IJ.selectWindow(window)
		IJ.run("Close")
		
		
