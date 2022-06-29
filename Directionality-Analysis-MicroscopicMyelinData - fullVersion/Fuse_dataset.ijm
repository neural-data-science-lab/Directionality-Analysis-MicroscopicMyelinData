// author: Gesine Müller
// only Fuse dataset step of BigStitcher

path = "/home/muellerg/nas/Gesine_Muellerg/PR012_11-53-16/";
name = "PR012_11-53-16";

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
    "export_path=" + path + name + "-f02");