// Pipeline to create: data before vesselness, mask_otsu from myelin channel and mask_cortex from autofluorescence channel

path = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Testdatensatz-0504/";
name_myelin = "C03.tif";
name_auto = "C00.tif";

open(path + name_myelin);
width = getWidth();
height = getHeight();
depth = nSlices;

Stack.setXUnit("um");
Stack.setYUnit("um");
Stack.setZUnit("um");
run("Properties...", "channels=1 slices=50 frames=1 pixel_width=0.542 pixel_height=0.542 voxel_depth=4");

// apply Gaussian filter and background substraction
run("Gaussian Blur...", "sigma=2 stack");
run("Subtract Background...", "rolling=95 stack");
saveAs("Tiff", path + name_myelin + "_smooth2_bg95.tif");

// apply otsu
setAutoThreshold("Otsu dark");
run("Convert to Mask", "method=Otsu background=Dark calculate black");
run("Fill Holes", "stack"); 
saveAs("Tiff", path + name_myelin + "_smooth2_bg95_otsu.tif");
run("Close");

// create mask_cortex
open(path + name_auto);
Stack.setXUnit("um");
Stack.setYUnit("um");
Stack.setZUnit("um");
run("Properties...", "channels=1 slices=50 frames=1 pixel_width=0.542 pixel_height=0.542 voxel_depth=4");
run("Gaussian Blur...", "sigma=45 stack");
setAutoThreshold("Otsu dark");
run("Convert to Mask", "method=Otsu background=Dark calculate black");
run("Fill Holes", "stack"); 
saveAs("Tiff", path + name_auto + "_cortex.tif");
run("Close");