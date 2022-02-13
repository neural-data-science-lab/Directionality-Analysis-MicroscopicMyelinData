// Pipeline to create: data before vesselness, mask_otsu from myelin channel and mask_cortex from autofluorescence channel

// read dataset path, number of tiles as commandline arguments
args = getArgument()
args = split(args, " ");
path = args[0];
if (!endsWith(path, File.separator))
{
    path = path + File.separator;
}
name = args[1];


run("Scriptable load HDF5...", "load="+path+name+".h5 datasetnames=/t00000/s03/0/cells nframes=1 nchannels=1");
//open(path + name + "_C03_gauss.tif");
run("Properties...", "pixel_width=0.5417 pixel_height=0.5417 voxel_depth=6");
width = getWidth();
height = getHeight();
depth = nSlices;

// apply Gaussian filter and background substraction
run("Gaussian Blur...", "sigma=2 stack");
saveAs("Tiff", path + name + "_"+ "C03_gauss.tif");
run("Subtract Background...", "rolling=95 stack");
saveAs("Tiff", path + name + "_"+ "C03_bg.tif");

// apply otsu
setAutoThreshold("Otsu dark");
run("Convert to Mask", "method=Otsu background=Dark calculate black");
saveAs("Tiff", path + name +"_"+ "C03_otsu.tif");
close();

// create mask_cortex
run("Scriptable load HDF5...", "load="+path+name+".h5 datasetnames=/t00000/s00/0/cells nframes=1 nchannels=1");
run("Properties...", "pixel_width=0.5417 pixel_height=0.5417 voxel_depth=6");
//run("Gaussian Blur...", "sigma=2 stack");
setAutoThreshold("Otsu dark");
run("Convert to Mask", "method=Otsu background=Dark calculate black");
run("Options...", "iterations=1 count=1 black do=Nothing");
run("Fill Holes", "stack");
saveAs("Tiff", path + name +"_"+ "C00_otsu.tif");
close();