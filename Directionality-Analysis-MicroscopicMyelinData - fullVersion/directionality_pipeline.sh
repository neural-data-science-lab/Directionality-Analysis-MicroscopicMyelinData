#!/bin/bash
# author: Gesine Müller
# Shell file for cropped x-y datasets
name"PR012_l_ACx"
path="/ptmp/muellerg"
patch_size=37
batch_size=20

conda activate imagej
# pre-proessing (Fiji: cortex, otsu, bg, py: vesselness filter)
#~/Downloads/fiji-linux64/Fiji.app/./ImageJ-linux64 --console -macro fiji_Pipeline.ijm "$path/$name/ $name"
# optimize cortex mask
#python optimize_cortexMask $name $path
# vesselness filter
#python processing_pipeline_headless.py $name $path/$name/ $patch_size $batch_size False True False False False

# Directionality Plugin
#~/Downloads/fiji-linux64/Fiji.app/./ImageJ-linux64 --ij2 --console --run fiji_Directionality.py 'patch_size=$patch_size filename="$name_C03_frangi.tif" path="$path/$name/" name=$name'

# Analysis (processing_pipeline (2x: for Directionality Plugin and for OrientationJ))
#python processing_pipeline_headless.py $name $path/$name/ $patch_size $batch_size False False False True True  
#python processing_pipeline_headless.py $name $path/$name/ $patch_size $batch_size False False True False True  