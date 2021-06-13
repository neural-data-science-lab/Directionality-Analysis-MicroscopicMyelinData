import scipy.signal as sig
import scipy.spatial as spa
import seaborn as sns #statistical data visualization
from skimage import measure
from skimage.feature import peak_local_max
from slab_microscopy import *

sns.set()
sns.set_style("white")
sns.despine()

def return_sparse(array,ratio):
    '''
    Turns a 2d array into a sparse representation of the same array. 
    Returns only the x-th element in the array along the first axis.
    (e.g. array with shape [200,3] with ratio 10 becomes array of shape [20,3])
    converts the array to float32
    '''
    if len(array)%ratio==0: #if theres no even division possible, elongate array by 1
        sparse_array=np.zeros((len(array)//ratio,array.shape[1]))
    if len(array)%ratio!=0:
        sparse_array=np.zeros((len(array)//ratio+1,array.shape[1]))

    sparse_array=sparse_array.astype("float32")

    for i in range(len(array)):
        if i%ratio==0:#only put every ratio-th coordinate into sparse version of array
            sparse_array[i//ratio]=array[i]

    return sparse_array

def xyz_to_zyx(array):
    '''
    swaps first and third row along the second axis of an array. e.g. turns an XYZ array to a ZYX array.
    '''
    temp=np.zeros((len(array)))
    temp=np.copy(array[:,0])#store X coords in temp array
    array[:,0]=array[:,2]#write Z coords where X are
    array[:,2]=temp#write X coords where Z are
    return array

def rotate_to_surface(cellcoords,n_cell,surface_norms,surface_verts):
    '''
    Rotate a dataset with the shape (n,3). Rotation is done so the new Z axis is orthogonal to the cortex surface.
    cellcoords:     int (not uint) input array with shape (n,3) with n Being the number of cells and the second axis being xyz coordinates (in that order).
    n_cell:         position of the "active" cell, which will be the origin of the rotation.
    surface_verts:  array of coordinates of the cortex surface, in the same space as cellcoords.
    surface_norms:  array of surface vectors corresponding to surface_verts
    '''
    ## define rotation points
    a=cellcoords[n_cell]#active cell

    #convert subset of coord_tree to a cKDTree
    coord_tree=spa.cKDTree(surface_verts)

    #find nearest neighbor to active cell in surface coords
    idx_nearest_surface = coord_tree.query(a,k=1)[1]

    # new Z is a + normal vector on the surface
    # b=a.astype("float64")+surface_norms[idx_nearest_surface]

    # v_ab=b-a #vector from a to b

    v_ab=surface_norms[idx_nearest_surface]

    #move set so that active cell is at origin (0,0,0)
    cellcoords=cellcoords-a

    ###find euler angles by reducing one dimension at each time, calculating the respective angle to turn around the axis at a time
    #gamm: angle between xy subvector and Y Axis -- rotation around Z axis
    v_y=np.array([0,1])#2d y vector [x,y]
    v_ab_xy=v_ab[0:2] #reduce 3d vector to XY plane
    gamm=m.acos((np.dot(v_ab_xy,v_y))/(m.sqrt(v_ab_xy[0]*v_ab_xy[0]+v_ab_xy[1]*v_ab_xy[1])*m.sqrt(v_y[0]*v_y[0]+v_y[1]*v_y[1])))    #calculate angle between v_ab_xy and y vector
    gamm=m.degrees(gamm)#angle to turn around Z axis

    #alph: angle between yz subvector and Z Axis -- rotation around X axis
    v_z=np.array([0,1])#2d z vector [y,z]
    v_ab_yz=v_ab[1:3] #reduce vector to YZ plane
    alph=m.acos((np.dot(v_ab_yz,v_z))/(m.sqrt(v_ab_yz[0]*v_ab_yz[0]+v_ab_yz[1]*v_ab_yz[1])*m.sqrt(v_z[0]*v_z[0]+v_z[1]*v_z[1])))    #calculate angle between v_ab_yz and Z vector
    alph=m.degrees(alph) #angle to turn around X axis

    #turn the dataset accordingly
    rot = R.from_euler("zyx",(gamm,0,alph),degrees=True)#initialize rotation
    coord_rotated = rot.apply(cellcoords) #rotate
    return coord_rotated, idx_nearest_surface, rot 

plot=False
save=True
n_cell=25 #number of cell to rotate around

# open raw images
af_raw = tf.imread(r"F:\CS002\subset_small\CS002_C00_00x00_Z200_Z250.tif")
hucd_raw = tf.imread(r"F:\CS002\subset_small\CS002_C01_00x00_Z200_Z250.tif")

# create gabor shell
shell=gabor_shell(radius=8,freq=0.08,sigma=5.84,z_scale_factor=6)

print("convolving .. ")
# convolve gabor shell on raw data
gabor_img=sig.fftconvolve(hucd_raw,shell,mode="same")

# adjusting img intensity
gabor_img = gabor_img*ndi.filters.gaussian_filter(hucd_raw, sigma=0.7)
gabor_img = gabor_img.astype("float32")

print("Maxima detection ..  ")
# maxima detection
centers = peak_local_max(
                        image=gabor_img,
                        min_distance=4,
                        indices=False,
                        footprint=morph.ball(4),
                        exclude_border=0,
                        )

# forget maxima that are not in the tissue
centers[hucd_raw<2000]=0

# insert maxima into gaussian
hucd_raw_centers = np.copy(hucd_raw)
hucd_raw_centers[centers==True] = 65535

# optional: save
if save==True:
    tf.imsave(file=r"F:\Philip\claudius_crops\raw_cellcenters.tif",data=hucd_raw_centers)
    tf.imsave(file=r"F:\Philip\claudius_crops\hucd_cellcenters.tif",data=centers)

print("swap Axes ..  ")
#swap axes so array has the same dimensions as marching cubes output
cellcoords = np.swapaxes(np.array(np.nonzero(centers)),0,1) 

print("Perform Marching Cubes ..  ")
verts, faces, norm, values = measure.marching_cubes_lewiner(volume=af_raw,level=2000,step_size=20, spacing=(20,20,20), allow_degenerate=False)#1200
#normalize verts to same dimensions as cellcenters
verts_scaled=verts.astype("uint32")//20

#make xyz to zyx
cellcoords=xyz_to_zyx(cellcoords)
verts_scaled=xyz_to_zyx(verts_scaled)

#normalize Z anisotropy
cellcoords[:,2]=cellcoords[:,2]*12.92 #7um/0.542um = 12.92
verts_scaled[:,2]=verts_scaled[:,2]*12.92 #7um/0.542um = 12.92

if plot==True: #plot vertices as smooth surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],cmap='Spectral', lw=1)
    plt.show()

### plot sparse version of cell centers:
sparse_cellcoords=return_sparse(cellcoords,100)
sparse_verts=return_sparse(verts_scaled,40)
sparse_norms=return_sparse(norm,40)

#perform rotation
cells_rotated, idx_surf, rot = rotate_to_surface(cellcoords=sparse_cellcoords,n_cell=n_cell,surface_norms=sparse_norms,surface_verts=sparse_verts)

if plot == True:#plot before rotation
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(sparse_cellcoords[:,0], sparse_cellcoords[:,1], sparse_cellcoords[:,2],marker=".") #before rotation
    ax.scatter(sparse_verts[:,0], sparse_verts[:,1], sparse_verts[:,2],marker="o") #verts before rotation
    ax.quiver(sparse_verts[:,0], sparse_verts[:,1], sparse_verts[:,2],sparse_norms[:,0], sparse_norms[:,1], sparse_norms[:,2], length=80,normalize=True,arrow_length_ratio=0.5,color="orange")
    ax.text(sparse_cellcoords[n_cell,0],sparse_cellcoords[n_cell,1],sparse_cellcoords[n_cell,2], '%s' % ("cell"), size=10, zorder=1,)#active cell
    ax.text(sparse_verts[idx_surf,0],sparse_verts[idx_surf,1],sparse_verts[idx_surf,2], '%s' % ("surface"), size=10, zorder=1,)#surface cell
    ax.set_zlabel("Z")
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    plt.show()

if plot == True: #plot after rotation
    labels_r=("cell_r","surface_r")
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(cells_rotated[:,0], cells_rotated[:,1], cells_rotated[:,2]) #after rotation cells
    ax.text(cells_rotated[n_cell,0],cells_rotated[n_cell,1],cells_rotated[n_cell,2], '%s' % (labels_r[0]), size=10, zorder=1,)#active cell
    ax.set_zlabel("Z")
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    plt.show()

## code graveyard

# distances,indices = coord_tree.query(pt,k=1) #query 1 point. find nearest neighbor and return distance and index of point
# pt=[1,9,200]#point to query
# nearest = cellcoords[spa.KDTree(cellcoords).query(pt)[1]]#returns index of nearest point in cellcoords