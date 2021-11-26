// Pipeline to create: data before vesselness, mask_otsu from myelin channel and mask_cortex from autofluorescence channel

// read data
//side = "Left";
//path = "E:/MasterThesis/PR012_l/AC_z275-625/";
//name = "PR012_l-ACx";

// read dataset path, number of tiles as commandline arguments
args = getArgument()
args = split(args, " ");
side = args[0];
path = args[1];
if (!endsWith(path, File.separator))
{
    path = path + File.separator;
}
name = args[2];


run("Scriptable load HDF5...", "load="+path+name+".h5 datasetnames=/t00000/s03/0/cells nframes=1 nchannels=1");
//open(path + side + name + "_C03_gauss.tif");
run("Properties...", "pixel_width=0.5417 pixel_height=0.5417 voxel_depth=6");
width = getWidth();
height = getHeight();
depth = nSlices;

// apply Gaussian filter and background substraction
run("Gaussian Blur...", "sigma=2 stack");
//run("Subtract Background...", "rolling=95 stack");
saveAs("Tiff", path + side + name + "_"+ "C03_bg.tif");

// apply otsu
setAutoThreshold("Otsu dark");
run("Convert to Mask", "method=Otsu background=Dark calculate black");
// run("Make Binary", "method=Default background=Dark calculate black");
saveAs("Tiff", path + side + name +"_"+ "C03_otsu.tif");
close();

// create mask_cortex
//run("Scriptable load HDF5...", "load="+path+name+".h5 datasetnames=/t00000/s00/1/cells nframes=1 nchannels=1");
open(path + side + name + "_C00_gauss.tif");
run("Properties...", "pixel_width=0.5417 pixel_height=0.5417 voxel_depth=6");
//run("Gaussian Blur...", "sigma=35 stack");
setAutoThreshold("Huang dark");
run("Convert to Mask", "method=Huang background=Dark calculate black");
run("Fill Holes", "stack");
saveAs("Tiff", path + side + name +"_"+ "C00_cortex.tif");
close();