# Does not work so far!!!
from ij import IJ, Prefs
from ij import WindowManager, ImagePlus

filename = "test_C00.tif"
path = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/"
outputpath = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03_smooth3D_bg95/"
name = "C00_radius_z"
img = IJ.openImage(path + filename)
width = img.getDimensions()[0]
height = img.getDimensions()[1]
depth = img.getDimensions()[3]

#for i in range(2):
IJ.setSlice(1)
IJ.run("Duplicate...", " ")
##def radius_slice(slice):
IJ.run("Options...", "black")
IJ.setAutoThreshold(img, "Intermodes dark")
IJ.run("Convert to Mask")
h*=0.5
IJ.doWand(0,h)
IJ.run("Interpolate", "interval="+h)

x = []
y = []
IJ.getSelectionCoordinates(xx, yy)
rank = IJ.Array.rankPositions(xx)
for i in range(3):
	x.append(xx[rank[i]])
	y.append(yy[rank[i]])

#code from <https://bitbucket.org/davemason/threepointcircumcircle/src> 
d1=((x[0]-x[1])*(x[0]-x[1])+(y[0]-y[1])*(y[0]-y[1]))**(1/2)
d2=((x[1]-x[2])*(x[1]-x[2])+(y[1]-y[2])*(y[1]-y[2]))**(1/2)
d3=((x[2]-x[0])*(x[2]-x[0])+(y[2]-y[0])*(y[2]-y[0]))**(1/2)
r=(d1*d2*d3)/((d1+d2+d3)*(d2+d3-d1)*(d3+d1-d2)*(d1+d2-d3))**(1/2)
print(r)
	#return r


		
#for i in range(2):
	#slice = IJ.setSlice(i)
	#l.append(radius_slice(slice))
	#print(l)


