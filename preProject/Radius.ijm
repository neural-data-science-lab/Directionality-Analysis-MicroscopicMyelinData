//Macro to compute the radius of the cortex, assuming a circle
// from: https://forum.image.sc/t/how-to-calculate-the-radius-of-curvature-for-an-image-of-an-arc/21341/4
//open("C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C00_1slice.tif");
setOption("BlackBackground", true);
w=getWidth();
h=getHeight();
//makeRectangle(0,0,w,h);
//run("Crop");
setAutoThreshold("Intermodes dark");
run("Convert to Mask");
h*=0.5;
doWand(0,h);
run("Interpolate", "interval="+h);
x = newArray(3);
y = newArray(3);
getSelectionCoordinates(xx, yy);
rank = Array.rankPositions(xx);
for ( i=0; i<3; i++ ) { 
   x[i] = xx[rank[i]];
   y[i] = yy[rank[i]];
}
/* the following code is taken from
<https://bitbucket.org/davemason/threepointcircumcircle/src> */
d1=sqrt((x[0]-x[1])*(x[0]-x[1])+(y[0]-y[1])*(y[0]-y[1]));
d2=sqrt((x[1]-x[2])*(x[1]-x[2])+(y[1]-y[2])*(y[1]-y[2]));
d3=sqrt((x[2]-x[0])*(x[2]-x[0])+(y[2]-y[0])*(y[2]-y[0]));
r=(d1*d2*d3)/sqrt((d1+d2+d3)*(d2+d3-d1)*(d3+d1-d2)*(d1+d2-d3));
print("Radius: "+d2s(r, 2));
//close();

