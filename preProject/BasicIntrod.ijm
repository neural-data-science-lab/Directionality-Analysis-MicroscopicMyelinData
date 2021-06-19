// open file
imageFilename = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03.tif";
open ( imageFilename );
width = getWidth();
height = getHeight();
getPixelSize(unit, pixelWidth, pixelHeight);
getStatistics(area, mean, min, max, std, histogram);

// first visual introduction to dataset
run("Histogram", "stack");
saveAs("Results", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03_histogram.csv");
close();

// set pixel size
run("Set Measurements...", "area mean standard modal min centroid perimeter fit shape feret's integrated median skewness kurtosis redirect=None decimal=3");
run("Measure");
saveAs("Results", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03_Results.csv");
Stack.setXUnit("um");
Stack.setYUnit("um");
Stack.setZUnit("um");
run("Properties...", "channels=1 slices=50 frames=1 pixel_width=0.542 pixel_height=0.542 voxel_depth=4");

//Filter
open("C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03.tif");
run("Gaussian Blur 3D...", "x=2 y=2 z=2");
saveAs("Tiff", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03_gaussian3D.tif");

run("Histogram", "stack");
saveAs("Results", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03_gaussian3D_histogram.csv");
close();

// Background correction
run("Subtract Background...", "rolling=95 stack");
saveAs("Tiff", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03_gaussian3D_bg95.tif");

setAutoThreshold("Otsu dark");
setAutoThreshold("Otsu dark stack");
run("Convert to Mask", "method=Otsu background=Dark calculate black");
run("Close");
saveAs("Tiff", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/test/test_C03_gaussian3D_bg95_3Dotsu.tif");

// 10.06
open("C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Datensatz-0705/Left_[01x00]_Z500_Z1000.tif");
//run("Brightness/Contrast...");
run("Enhance Contrast", "saturated=0.35");
run("Apply LUT", "stack");
run("Close");

Stack.setXUnit("um");
Stack.setYUnit("um");
Stack.setZUnit("um");
run("Properties...", "channels=1 slices=501 frames=1 pixel_width=0.542 pixel_height=0.542 voxel_depth=4");
run("Gaussian Blur...", "sigma=2 stack");
saveAs("Tiff", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Datensatz-0705/Left_gaussian.tif");
run("Subtract Background...", "rolling=95 stack");
saveAs("Tiff", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Datensatz-0705/Left_gaussian_bg95.tif");
setAutoThreshold("Mean dark");
//run("Threshold...");
setAutoThreshold("Otsu dark");
run("Close");
saveAs("Tiff", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Datensatz-0705/Left_gaussian_bg95_otsu.tif");

//16.06 for 50 slices 
Stack.setXUnit("um");
Stack.setYUnit("um");
Stack.setZUnit("um");
run("Properties...", "channels=1 slices=50 frames=1 pixel_width=0.542 pixel_height=0.542 voxel_depth=4");
run("Select All");
run("Save");
run("Gaussian Blur...", "sigma=2 stack");
run("Subtract Background...", "rolling=95 stack");
//run("Brightness/Contrast...");
run("Enhance Contrast", "saturated=0.35");
run("Apply LUT", "stack");
run("Close");
saveAs("Tiff", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Datensatz-0705/RightZ50_smooth2_bg95.tif");
setAutoThreshold("Otsu dark");
//run("Threshold...");
setOption("BlackBackground", true);
run("Convert to Mask", "method=Otsu background=Dark calculate black");
run("Close");
run("Open", "stack");
saveAs("Tiff", "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Datensatz-0705/RightZ50_smooth2_bg95_otsu.tif");

// for cortex mask -> autofluorescence channel
run("Duplicate...", "duplicate");
run("Gaussian Blur...", "sigma=45 stack");
run("Close");
setAutoThreshold("Otsu dark");
run("Convert to Mask", "method=Otsu background=Dark calculate black");
run("Fill Holes", "stack"); 