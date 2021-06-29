// Pipeline to create: data before vesselness, mask_otsu from myelin channel and mask_cortex from autofluorescence channel

// ToDo: change side
side = "Right";


path = "/media/muellerg/Data SSD/Gesine/Data/";
name_myelin = side + "_[01x07]_Z500_Z1000.tif";
name_auto = side + "_AF.tif";

open(path + name_myelin);
width = getWidth();
height = getHeight();
depth = nSlices;

Stack.setXUnit("um");
Stack.setYUnit("um");
Stack.setZUnit("um");
run("Properties...", "channels=1 slices=depth frames=1 pixel_width=0.542 pixel_height=0.542 voxel_depth=4");

// apply Gaussian filter and background substraction
run("Gaussian Blur...", "sigma=2 stack");
run("Subtract Background...", "rolling=95 stack");
run("Bio-Formats Exporter", "save=path + side + _smooth2_bg95.tif compression=Uncompressed");
//saveAs("Tiff", path + side + "_smooth2_bg95.tif");

// apply otsu
setAutoThreshold("Otsu dark");
run("Convert to Mask", "method=Otsu background=Dark calculate black");
run("Fill Holes", "stack");
run("Bio-Formats Exporter", "save=path + side + _smooth2_bg95_otsu.tif compression=Uncompressed");
//saveAs("Tiff", path + side + "_smooth2_bg95_otsu.tif");
close();

// create mask_cortex
open(path + name_auto);
Stack.setXUnit("um");
Stack.setYUnit("um");
Stack.setZUnit("um");
run("Properties...", "channels=1 slices=depth frames=1 pixel_width=0.542 pixel_height=0.542 voxel_depth=4");
run("Gaussian Blur...", "sigma=45 stack");
setAutoThreshold("Triangle dark");
run("Convert to Mask", "method=Triangle background=Dark calculate black");
run("Fill Holes", "stack");
run("Bio-Formats Exporter", "save=path + side + _cortex.tif compression=Uncompressed"); 
//saveAs("Tiff", path + side + "_cortex.tif");
close();
