// Run BigStitcher in headless mode
args = getArgument()
args = split(args, " ");

path = args[0];
if (!endsWith(path, File.separator))
{
    path = path + File.separator;
}
name = args[1];


// define dataset
//run("Define dataset ...",
//    "define_dataset=[Automatic Loader (Bioformats based)]" +
//    " project_filename="+name+".xml path=" + path + "*.tif exclude=10" +
//    " bioformats_channels_are?=Channels pattern_0=Tiles pattern_1=Tiles modify_voxel_size? voxel_size_x=0.541667" +
//    " voxel_size_y=0.541667 voxel_size_z=6 voxel_size_unit=um " +
//    "move_tiles_to_grid_(per_angle)?=[Move Tile to Grid (Macro-scriptable)] grid_type=[Grid: row-by-row      ]" +
//    " tiles_x=8 tiles_y=3 tiles_z=1 overlap_x_(%)=20 overlap_y_(%)=20 overlap_z_(%)=20" +
//    " keep_metadata_rotation how_to_load_images=[Re-save as multiresolution HDF5] " +
//    "dataset_save_path=" + path +
//    "subsampling_factors=[{ {1,1,1}, {2,2,1}, {4,4,1}, {8,8,1}, {16,16,2}, {32,32,4}, {64,64,8} }] " + 
//    "hdf5_chunk_sizes=[{ {32,32,4}, {32,32,4}, {32,16,8}, {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16} }] " +
//    "timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression " +
//    "export_path=" + path + name);

// calculate pairwise shifts
run("Calculate pairwise shifts ...",
    "select="+path+name+".xml process_angle=[All angles] process_channel=[All channels]" +
    " process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints]" +
    " method=[Phase Correlation] channels=[Average Channels] downsample_in_x=4 downsample_in_y=4 downsample_in_z=1");

// filter shifts with 0.7 corr. threshold
run("Filter pairwise shifts ...",
    "select="+path+name+".xml filter_by_link_quality min_r=0.7 max_r=1 " +
    "max_shift_in_x=0 max_shift_in_y=0 max_shift_in_z=0 max_displacement=0");

// do global optimization
run("Optimize globally and apply shifts ...",
    "select="+path+name+".xml process_angle=[All angles] process_channel=[All channels] " +
    "process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints]" +
    " relative=2.500 absolute=3.500 global_optimization_strategy=" +
    "[Two-Round using Metadata to align unconnected Tiles] fix_group_0-0,");


// fuse dataset, save as hdf5
run("Fuse dataset ...",
    "select="+path+name+".xml process_angle=[All angles] process_channel=[All channels] " +
    "process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints]" + 
    " bounding_box=[All Views] downsampling=1 pixel_type=[16-bit unsigned integer] interpolation=[Linear Interpolation]" +
    " image=[Cached] interest_points_for_non_rigid=[-= Disable Non-Rigid =-] blend preserve_original " +
    "produce=[Each timepoint & channel] fused_image=[Save as new XML Project (HDF5)] " +
    "subsampling_factors=[{ {1,1,1}, {2,2,1}, {4,4,1}, {8,8,1}, {16,16,2}, {32,32,4}, {64,64,8} }] " +
	"hdf5_chunk_sizes=[{ {32,32,4}, {32,32,4}, {32,16,8}, {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16} }] timepoints_per_partition=1 " +
	"setups_per_partition=0 use_deflate_compression " +
    "export_path=" + path + name + "-f0");


// quit after we are finished
eval("script", "System.exit(0);");