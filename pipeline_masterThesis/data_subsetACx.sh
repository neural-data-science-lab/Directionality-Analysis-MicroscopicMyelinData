#!/bin/bash
# Shell file for cropped x-y datasets (no Stitching, no Registration step)
nameIn="PR014_l"
nameOut="PR014_l_ACx"

# crop in z-direction
cd "/home/muellerg/nas/Philip_Ruthig/5_Philip_PhD_Data_Mice/new_substacks_gesine/$nameIn"
mkdir "/home/muellerg/nas/Gesine_Muellerg/new_substacks_gesine/$nameOut"
ls |  grep Z0[2][7][5-9].ome* | xargs -I '{}' cp '{}'  ~/nas/Gesine_Muellerg/new_substacks_gesine/$nameOut
ls |  grep Z0[2][8-9][0-9].ome* | xargs -I '{}' cp '{}'  ~/nas/Gesine_Muellerg/new_substacks_gesine/$nameOut
ls |  grep Z0[3-5][0-9][0-9].ome* | xargs -I '{}' cp '{}'  ~/nas/Gesine_Muellerg/new_substacks_gesine/$nameOut
ls |  grep Z0[6][0-1][0-9].ome* | xargs -I '{}' cp '{}'  ~/nas/Gesine_Muellerg/new_substacks_gesine/$nameOut
ls |  grep Z0[6][2][0-5].ome* | xargs -I '{}' cp '{}'  ~/nas/Gesine_Muellerg/new_substacks_gesine/$nameOut

cd "/home/muellerg/nas/Gesine_Muellerg"
# resave datset as hdf5
~/Downloads/fiji-linux64/Fiji.app/./ImageJ-linux64 --headless --console -macro HDF5_defineDataset.ijm "$nameOut /home/muellerg/nas/Gesine_Muellerg/new_substacks_gesine/$nameOut/"