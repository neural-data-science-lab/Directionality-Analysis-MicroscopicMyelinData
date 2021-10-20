// Pipeline to create: data before vesselness, mask_otsu from myelin channel and mask_cortex from autofluorescence channel

// read data
side = "Left";
path_input = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/";
path_output = "C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/Analyse_Directionality/Testdatensatz-0504/test/Pipeline/";
name_myelin = "C03.tif";
name_auto = "C00.tif";

open(path_input + name_myelin);
width = getWidth();
height = getHeight();
depth = nSlices;

// apply Gaussian filter and background substraction
run("Gaussian Blur...", "sigma=2 stack");
run("Subtract Background...", "rolling=95 stack");
//run("Bio-Formats Exporter", "save=path_output + side + C03_bg.tif compression=Uncompressed");
saveAs("Tiff", path_output + side +"_"+ "C03_bg.tif");

// apply otsu
setAutoThreshold("Otsu dark");
run("Convert to Mask", "method=Otsu background=Dark calculate black");
//run("Bio-Formats Exporter", "save=path_output + side + C03_otsu.tif compression=Uncompressed");
saveAs("Tiff", path_output + side +"_"+ "C03_otsu.tif");
close();

// create mask_cortex
open(path_input + name_auto);
run("Gaussian Blur...", "sigma=35 stack");
setAutoThreshold("Huang dark");
run("Convert to Mask", "method=Huang background=Dark calculate black");
run("Fill Holes", "stack");
//run("Bio-Formats Exporter", "save=path_output + side + C00_cortex.tif compression=Uncompressed");
saveAs("Tiff", path_output + side +"_"+ "C00_cortex.tif");
close();
