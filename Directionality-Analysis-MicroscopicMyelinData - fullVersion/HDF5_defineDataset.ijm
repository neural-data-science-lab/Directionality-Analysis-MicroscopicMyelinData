// author: Gesine Müller
// create hdf5 file of a folder of tiffs

//Run Define dataset in headless mode -> results in hdf5

// read variables as commandline arguments

// read dataset path, number of tiles as commandline arguments
args = getArgument()
args = split(args, " ");
name = args[0];
path = args[1];
if (!endsWith(path, File.separator))
{
    path = path + File.separator;
}

run("Define dataset ...", 
	"define_dataset=[Automatic Loader (Bioformats based)] project_filename="+name+".xml path=" + path + "*.tif exclude=10 " + 
	"pattern_0=Channels pattern_1=[Z-Planes (experimental)] modify_voxel_size? voxel_size_x=0.541667" +
    " voxel_size_y=0.541667 voxel_size_z=6 voxel_size_unit=um " +
    "move_tiles_to_grid_(per_angle)?=[Move Tiles to Grid (interactive)] "+
	"how_to_load_images=[Re-save as multiresolution HDF5] dataset_save_path=" + path + " manual_mipmap_setup " +
	"subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8}, {16,16,16} }] hdf5_chunk_sizes=[{ {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16} }] " +
	"timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression export_path=" + path + name);