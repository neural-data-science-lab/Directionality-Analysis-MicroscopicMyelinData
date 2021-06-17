from ij import IJ, Prefs
from ij import WindowManager, ImagePlus
from fiji.analyze.directionality import Directionality_

filename = "RightZ50_smooth2_bg95_sato.tif"
path = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Datensatz-0705/"
outputpath = path+"RightZ50_smooth2_bg95_sato_dice20/"
name = "rightDice20_"
img = IJ.openImage(path + filename)

width = img.getDimensions()[0]
height = img.getDimensions()[1]
if ( (width % 20) != 0  ): print("Adjust patch size.")
if ( (height % 20) != 0  ): print("Adjust patch size.")
x_bound = width/20
y_bound = height/20

#create patches and run directionality on them
for i in range(x_bound):
	for j in range(y_bound):
		img2 = img.duplicate()
		k = i*20
		l = j*20
		img2.setRoi(k, l, 20, 20)
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
		
		
