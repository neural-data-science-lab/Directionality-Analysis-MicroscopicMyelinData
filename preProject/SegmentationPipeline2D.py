""" Construction of a 2D single-cell segmentation pipeline
following: Image analysis tutorial https://github.com/WhoIsJack/python-bioimage-analysis-tutorial/blob/master/image_analysis_tutorial.ipynb

data: single-color spinning-disk confocal micrographs (objective: 40X 1.2NA W) of cells in live zebrafish embryos
in early development, fluorescently labelled """

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import os
import skimage.io as io
from skimage.filters.thresholding import threshold_otsu, threshold_local, try_all_threshold
from skimage.filters import rank
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy.stats import describe
from scipy.stats import linregress
from skimage.io import imsave
import pickle

# Loading and Handling Image data
filename = 'example_cells_1.tif'
path = 'example_data'
filepath = os.path.join(path, filename)
print(filepath)
img = io.imread(filepath)
print("Array is of type:", type(img))
print("Array has shape:", img.shape)
print("Values are of type:", img.dtype)

# Preprocessing
## Background: enhance signal-to-noise, improve structures e.g.  Deconvolution, 8-bit, Cropping,  Smoothing, Artifact, Background substraction
### Gaussian smoothing
sigma = 2
img_smooth = ndi.filters.gaussian_filter(img, sigma=sigma)

#fig, ax = plt.subplots(1, 2, figsize=(12, 8), dpi=120)
#ax[0].imshow(img, interpolation='none', cmap='gray')
#ax[1].imshow(img_smooth, interpolation='none', cmap='gray')
#ax[0].set_title('Raw Image')
#ax[1].set_title('Smoothed Image')
#plt.show()

# crop image
# plt.figure(figsize=(6,6))
# plt.imshow(img_smooth[400:600, 200:400], interpolation='none', cmap='gray')
# plt.title('Cropped Image')
# plt.show()

# Manual thresholding & adaptive threshold detection
threshold = 70
img_thresh = img_smooth > threshold
print(img_thresh.dtype)

fig, ax = plt.subplots(1, 2, figsize=(12, 8))
ax[0].imshow(img_smooth, interpolation='none', cmap='gray')
ax[1].imshow(img_thresh, interpolation='none', cmap='gray')
ax[0].set_title('Smoothed Image')
ax[1].set_title('Thresholded Membranes')
plt.show()

## Otsu's method
thresh = threshold_otsu(img_smooth)
img_thresh = img_smooth > thresh
#plt.figure(figsize=(7, 7))
#plt.imshow(img_thresh, interpolation='none', cmap='gray')
#plt.show()

#fig = try_all_threshold(img_smooth, figsize=(15, 15), verbose=False)
#plt.show()

## Adaptive thresholding: scikit-image: threshold_local, here: own version
i = 31
SE = (np.mgrid[:i, :i][0] - np.floor(i / 2)) ** 2 + (np.mgrid[:i, :i][1] - np.floor(i / 2)) ** 2 <= np.floor(
    i / 2) ** 2  # structural element
bg = rank.mean(img_smooth, selem=SE)
img_bg = img_smooth > bg

## Comparison to scikit:
block_size = 47
adaptive_thresh = threshold_local(img_smooth, block_size, offset=10)
img_adaptive = img_smooth > adaptive_thresh
fig, ax = plt.subplots(1, 3, figsize=(12, 8))
ax[0].imshow(img_smooth, interpolation='none', cmap='gray')
ax[1].imshow(img_thresh, interpolation='none', cmap='gray')
ax[2].imshow(img_adaptive, interpolation='none', cmap='gray')
ax[0].set_title('Smoothed Image')
ax[1].set_title('Adaptive thresholding manual')
ax[1].set_title('Adaptive thresholding scikit')
plt.show()

# Improving masks with binary morphology: erosion, dilation, closing, opening
img_holefilled = ~ndi.binary_fill_holes(~img_bg)
# long: img_holefilled = np.logical_not(ndi.binary_fill_holes(np.logical_not(img_thresh)))
# further morphological operations + new mask, padding with reflection
i = 15
SE = (np.mgrid[:i, :i][0] - np.floor(i / 2)) ** 2 + (np.mgrid[:i, :i][1] - np.floor(i / 2)) ** 2 <= np.floor(i / 2) ** 2
pad_size = i + 1
mem_padded = np.pad(img_holefilled, pad_size, mode='reflect')
img_final = ndi.binary_closing(mem_padded, structure=SE)
img_final = img_final[pad_size:-pad_size, pad_size:-pad_size]  # crop back
fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].imshow(img_smooth, interpolation='none', cmap='gray')
ax[1].imshow(img_final, interpolation='none', cmap='gray')
ax[0].set_title('Smoothed Image')
ax[1].set_title('Final Membrane Mask')
plt.show()

# Connected components labeling: preliminary  segmentation of cells
cell_labels, _ = ndi.label(~img_final)
plt.figure(figsize=(7, 7))
plt.imshow(cell_labels, interpolation='none', cmap='inferno')
plt.show()

# Cell segmentation by seeding and expansion
dist_trans = ndi.distance_transform_edt(~img_final)  # distance transform
dist_trans_smooth = ndi.filters.gaussian_filter(dist_trans, sigma=5)  # smoothen distance transform

fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].imshow(dist_trans, interpolation='none', cmap='gray')
ax[1].imshow(dist_trans_smooth, interpolation='none', cmap='gray')
ax[0].set_title('Distance transform')
ax[1].set_title('Smoothed distance transform')
plt.show()
### check from here on: seeds are not correctly placed
seeds = peak_local_max(dist_trans_smooth, indices=False, min_distance=10)
seeds_dil = ndi.filters.maximum_filter(seeds, size=10)  # dilate seeds

plt.figure(figsize=(7, 7))
plt.imshow(img_smooth, interpolation='none', cmap='gray')
plt.imshow(np.ma.array(seeds_dil, mask=seeds_dil == 0), interpolation='none', cmap='autumn')
plt.show()

seeds_labeled = ndi.label(seeds)[0]
seeds_labeled_dil = ndi.filters.maximum_filter(seeds_labeled, size=10)  # Expand a bit for visualization
plt.figure(figsize=(10,10))
plt.imshow(img_smooth, interpolation='none', cmap='gray')
plt.imshow(np.ma.array(seeds_labeled_dil, mask=seeds_labeled_dil==0), interpolation='none', cmap='prism')
plt.show()

## Watershed expansion: topological height profile
ws = watershed(img_smooth, seeds_labeled)
plt.figure(figsize=(10,10))
plt.imshow(img_smooth, interpolation='none', cmap='gray')
plt.imshow(ws, interpolation='none', cmap='prism', alpha=0.4)
plt.show()

# Postprocessing: removing cells at the image border
border_mask = np.zeros(ws.shape, dtype=np.bool)
border_mask = ndi.binary_dilation(border_mask, border_value=1)
clean_ws = np.copy(ws)
for cell_ID in np.unique(ws):
    cell_mask = ws == cell_ID
    cell_border_overlap = np.logical_and(cell_mask, border_mask)  # Overlap of cell mask and boundary mask
    total_overlap_pixels = np.sum(cell_border_overlap)  # Sum overlapping pixels
    if total_overlap_pixels > 0:
        clean_ws[cell_mask] = 0

for new_ID, cell_ID in enumerate(np.unique(clean_ws)[1:]):  # The [1:] excludes 0 from the list (background)!, re-labeling
    clean_ws[clean_ws==cell_ID] = new_ID+1

plt.figure(figsize=(7,7))
plt.imshow(img_smooth, interpolation='none', cmap='gray')
plt.imshow(np.ma.array(clean_ws, mask=clean_ws==0), interpolation='none', cmap='prism', alpha=0.4)
plt.show()

# Measurement and data analysis
## Additional mask: Identifying cell edges; here based on erosion
edges = np.zeros_like(clean_ws)  # same shape and type as clean_ws
for cell_ID in np.unique(clean_ws)[1:]: # 0:background
    cell_mask = clean_ws == cell_ID
    eroded_cell_mask = ndi.binary_erosion(cell_mask, iterations=1)
    edge_mask = np.logical_xor(cell_mask, eroded_cell_mask)
    edges[edge_mask] = cell_ID

plt.figure(figsize=(7,7))
plt.imshow(np.zeros_like(edges)[300:500, 300:500], cmap='gray', vmin=0, vmax=1)  # Simple black background
plt.imshow(np.ma.array(edges, mask=edges==0)[300:500, 300:500], interpolation='none', cmap='prism')
plt.show()

## Extracting quantitative measurements: use raw data for that, best pandas data frame but here simply dict
results = {"cell_id"      : [],
           "int_mean"     : [],
           "int_mem_mean" : [],
           "cell_area"    : [],
           "cell_edge"    : []}

for cell_id in np.unique(clean_ws)[1:]:
    # Mask the current cell and cell edge
    cell_mask = clean_ws == cell_id
    edge_mask = edges == cell_id

    results["cell_id"].append(cell_id)
    results["int_mean"].append(np.mean(img[cell_mask]))
    results["int_mem_mean"].append(np.mean(img[edge_mask]))
    results["cell_area"].append(np.sum(cell_mask))
    results["cell_edge"].append(np.sum(edge_mask))

for key in results.keys():
    print(key + ":", results[key][:5], '\n')

## Simple analysis and visualization
def print_summary(data):
    print( "  Mean:    {:7.2f}".format(np.mean(data))   )
    print( "  Stdev:   {:7.2f}".format(np.std(data))    )
    print( "  Max:     {:7.2f}".format(np.max(data))    )
    print( "  Min:     {:7.2f}".format(np.min(data))    )
    print( "  Median:  {:7.2f}".format(np.median(data)) )

for key in results.keys():
    print( '\n'+key )
    print_summary(results[key])
stat_summary = describe(results['int_mean'])

print( '\nscipy.stats.describe of int_mean' )
for key in stat_summary._asdict().keys():
    print( ' ', key+': ', stat_summary._asdict()[key] )
### boxplot,mean,mem_mean
plt.figure(figsize=(3,6))
plt.boxplot([results['int_mean'], results['int_mem_mean']],
            labels=['int_mean', 'int_mem_mean'],
            widths=0.6, notch=True)
plt.show()
### scatterplot of cell outline
plt.figure(figsize=(8,5))
plt.scatter(results["cell_area"], results["cell_edge"],
            edgecolor='k', s=30, alpha=0.5)
plt.xlabel("cell area [pxl^2]")
plt.ylabel("cell edge length [pxl]")
### null model: cells = circular
cell_area_range = np.linspace(min(results["cell_area"]), max(results["cell_area"]), num=100)
circle_circumference = 2 * np.pi * np.sqrt( cell_area_range / np.pi )
plt.plot(cell_area_range, circle_circumference, color='r', alpha=0.8)
plt.legend(['circles', 'data'], loc=2, fontsize=10)
plt.show()
### linear fit of membrane intensity over cell area
linfit = linregress(results["cell_area"], results["int_mem_mean"])
linprops = ['slope','interc','rval','pval','stderr']
for index,prop in enumerate(linprops):
    print( prop, '\t', '{:4.2e}'.format(linfit[index]) )
### lin fit not meaningful eventhough p-val is significant -> only means slope of fit is unlikely to be 0, r-value: low -> model not meaningful
### rather have look on effect size than p-value for large datasets
### Bias: oversegmentation poss.
x_vals = [min(results["cell_area"]), max(results["cell_area"])]
y_vals = [linfit[0] * x_vals[0] + linfit[1], linfit[0] * x_vals[1] + linfit[1]]
plt.figure(figsize=(8,5))
plt.scatter(results["cell_area"], results["int_mem_mean"],
            edgecolor='k', s=30, alpha=0.5)
plt.plot(x_vals, y_vals, color='red', lw=2, alpha=0.8)
plt.legend(["linear fit, Rsq={:4.2e}".format(linfit[2]**2.0)], frameon=False, loc=4)
plt.xlabel("cell area [pxl]")
plt.ylabel("Mean membrane intensity [a.u.]")
plt.title("Scatterplot with linear fit")
plt.show()
### Map the cell area back onto the image as a 'heatmap'
areas_8bit = np.array(results["cell_area"]) / max(results["cell_area"]) * 255
area_map = np.zeros_like(clean_ws, dtype=np.uint8)
for index, cell_id in enumerate(results["cell_id"]):
    area_map[clean_ws == cell_id] = areas_8bit[index]
outlier_mask = np.logical_or(area_map > np.percentile(areas_8bit, 95),
                             area_map < np.percentile(areas_8bit, 5))
full_mask = np.logical_or(area_map==0, outlier_mask)
plt.figure(figsize=(10,10))
plt.imshow(img_smooth, interpolation='none', cmap='gray')
plt.imshow(np.ma.array(area_map, mask=full_mask),interpolation='none', cmap='viridis', alpha=0.6)
plt.show()

# Writing output into files
imsave("example_cells_1_edges.tif", edges.astype(np.uint16))

plt.scatter(results["cell_area"], results["int_mem_mean"], edgecolor='k', s=30, alpha=0.5)
plt.plot(x_vals, y_vals, color='red', lw=2, alpha=0.8)
plt.legend(["linear fit, Rsq={:4.2e}".format(linfit[2]**2.0)], frameon=False, loc=4)
plt.xlabel("cell area [pxl]")
plt.ylabel("Mean membrane intensity [a.u.]")
plt.title("Scatterplot with linear fit")
plt.savefig('example_cells_1_scatterFit.png')
plt.savefig('example_cells_1_scatterFit.pdf')
plt.clf()  # Clear the figure buffer

np.save("example_cells_1_seg", clean_ws)  # Save
seg = np.load("example_cells_1_seg.npy")  # Load
print(clean_ws.shape, seg.shape)

with open('example_cells_1_results.pkl','wb') as outfile:
    pickle.dump(results, outfile)
with open('example_cells_1_results.pkl', 'rb') as infile:
    results_reloaded = pickle.load(infile)
    print(results_reloaded.keys())
with open('example_cells_1_results.txt','w') as outfile:
    header_string = '\t'.join(results.keys()) + '\n'
    outfile.write(header_string)
    for index in range(len(results['cell_id'])):
        data_string = '\t'.join([str(results[key][index]) for key in results.keys()]) + '\n'
        outfile.write(data_string)

# Generate pipline
