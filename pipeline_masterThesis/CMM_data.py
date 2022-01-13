'''
construct a csv such that all result data is in one file - ready for statistics
1. add l/r and sample id
2. average over all z-slice with same layer_id and id for tonotopic axis
3. plot the densities of both and several conditions
'''

import os
import numpy as np
import pandas as pd
from scipy.stats import circmean
import seaborn as sns
import matplotlib.pyplot as plt


#### construct a csv such that all result data is in one file -> add l/r and sampleID
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/'
pixel = 0.5417
layers = np.array([0, 58.5, 234.65, 302.25, 557.05])/pixel
layer_ids = ['L1', 'L2/3', 'L4', 'L5', 'L6']

Result_Fiji = []
sample = ['09','12','14','17']
side = ['l', 'r']
for i in sample:
    for j in side:
        data=pd.read_csv(os.path.join(path, 'Result_PR0'+str(i)+'_'+str(j)+'_ACx_92_Fiji_Directionality_.csv'))
        side_id = np.array([str(j) for x in range(data.shape[0])])
        id = np.array([str(i) for x in range(data.shape[0])])
        data.insert(loc=0, column='side', value=side_id)
        data.insert(loc=0, column='id', value=id)
        layer_stack = []
        for k in range(data.shape[0]):
            layer_stack.append(layer_ids[np.digitize(data['4'][k], layers, right=False) - 1])
        layer_stack = np.vstack(layer_stack)
        data.insert(loc=2, column='layer', value=layer_stack)
        Result_Fiji.append(data)
Result_Fiji = pd.DataFrame(np.vstack(Result_Fiji))
Result_Fiji.to_csv(path+'Result_Fiji_92.csv', index=False)

Result_OriJ = []
for i in sample:
    for j in side:
        data=pd.read_csv(os.path.join(path, 'Result_PR0'+str(i)+'_'+str(j)+'_ACx_92_OrientationJ_.csv'))
        side_id = np.array([str(j) for x in range(data.shape[0])])
        id = np.array([str(i) for x in range(data.shape[0])])
        data.insert(loc=0, column='side', value=side_id)
        data.insert(loc=0, column='id', value=id)
        layer_stack = []
        for k in range(data.shape[0]):
            layer_stack.append(layer_ids[np.digitize(data['4'][k], layers, right=False) - 1])
        layer_stack = np.vstack(layer_stack)
        data.insert(loc=2, column='layer', value=layer_stack)
        Result_OriJ.append(data)
Result_OriJ = pd.DataFrame(np.vstack(Result_OriJ))
Result_OriJ.to_csv(path+'Result_OriJ_92.csv', index=False)


#### now restructure data: average over z-direction; output: sampleID, side, layer, y=tonotopicAxis, domDir
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_12141718/'
data_long = pd.read_csv(os.path.join(path, 'Result_Fiji_92.csv'))
layer_ids = ['L1', 'L2/3', 'L4', 'L5', 'L6']

result = []
for sampleId in [9,12,14,17]:
    for side in ['l', 'r']:
        data = data_long[(data_long['0'] == sampleId) & (data_long['1']== side)]
        for layer_id in layer_ids:
            for y in np.arange(int(data['4'].min()), int(data['4'].max())+1,1):
                z_sum = []
                for z in np.arange(int(data['3'].min()), int(data['3'].max())+1,1):
                    z_sum.append(np.array(data[(data['2'] == layer_id) & (data['4'] == y) & (data['3'] == z)]['6']))
                z_sum = np.deg2rad(np.concatenate(z_sum).ravel())
                z_counts = len(z_sum)
                #mean_circ = circmean(z_sum)
                mean_circ = pd.DataFrame(z_sum).mode()[0][0]
                result.append([sampleId, side, layer_id, y, mean_circ,z_counts])
Result = pd.DataFrame(result)
for i in range(len(Result[4])):
    if Result[4][i] > np.pi:
        Result[4][i] = Result[4][i] - 2. * np.pi
Result[4] = np.rad2deg(Result[4])
Result.to_csv(path+'Result_Fiji_92_mode-short.csv', index=False)



#### Plotting: density plots
patch_size = 92
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_12141718/'
Result_long = pd.read_csv(os.path.join(path, 'Result_Fiji_'+str(patch_size)+'.csv'))
Result_short = pd.read_csv(os.path.join(path, 'Result_Fiji_'+str(patch_size)+'-short.csv'))
Result_mode = pd.read_csv(os.path.join(path, 'Result_Fiji_'+str(patch_size)+'_mode-short.csv'))

#### density plot
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
#sns.set(font_scale = 1.5)
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.kdeplot(data=Result_long, x=Result_long['6'], bw=0.5, color="red")
#sns.kdeplot(Result_short['4'], bw=0.5, color="red")
sns.kdeplot(data=Result_long, x=Result_long['6'], hue=Result_long['2'], fill=True, common_norm=False,
            alpha=0.4, palette="Greys_r")
#sns.kdeplot(data=Result_short, x=Result_short['4'], hue=Result_short['2'], fill=True, common_norm=False,
#            alpha=0.4, palette="Greys_r", clip = (-90.0, 90.0))
ax1.set_ylabel('Density', fontsize=20)
ax1.set_xlabel('Dominant direction (°)', fontsize=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
ax1.set_xticklabels(ax1.get_xticks(),size=20)
plt.legend(labels=['Combined', 'L6','L5','L4','L2/3','L1'], title = '',
           fontsize = 'large', title_fontsize = '2', loc = 'upper center')
plt.show()

#### histogram + density plot
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
#sns.set(font_scale = 1.5)
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.histplot(data=Result_long, x=Result_long['6'], color="red", label="Combined", kde=True)
ax1.set_ylabel('Density', fontsize=20)
ax1.set_xlabel('Dominant direction (°)', fontsize=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
ax1.set_xticklabels(ax1.get_xticks(),size=20)
plt.legend(labels=['Combined'], title = '',
           fontsize = 20, title_fontsize = '2', loc = 'upper right')
plt.show()

############# all same for conditions
dat = Result_long[(Result_long['1']=='l') and (Result_long['2']!='L1')]

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
#sns.set(font_scale = 1.5)
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.kdeplot(dat['6'], bw=0.5, color="red")
#sns.kdeplot(dat['4'], bw=0.5, color="red")
sns.kdeplot(data=dat, x=dat['6'], hue=dat['2'], fill=True, common_norm=False, alpha=0.4, palette="Greys_r")
#sns.kdeplot(data=dat, x=dat['4'], hue=dat['2'], fill=True, common_norm=False, alpha=0.4, palette="Greys_r", clip = (-90.0, 90.0))
ax1.set_ylabel('Density', fontsize=20)
ax1.set_xlabel('Dominant direction (°)', fontsize=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
ax1.set_xticklabels(ax1.get_xticks(),size=20)
plt.legend(labels=['Combined', 'L6','L5','L4','L2/3','L1'], title = '',
           fontsize = 'large', title_fontsize = '2', loc = 'upper center')
plt.show()


'''dat = Result_short[(Result_short['2']!='L1')]
dat = dat[(dat['1']=='r')]
dat = dat[(dat['2']!='L6')]
dat = dat[(dat['0']!=18)]'''

dat = Result_long[(Result_long['3'].between(30,310))]
dat = dat[(dat['2']=='L2/3')]
dat = dat[(dat['4']==20)]

dat =  Result_short

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.histplot(data=dat, x=dat['4'], color="black", label="Combined", kde=True, bins=50)
ax1.set_ylabel('Density', fontsize=24)
ax1.set_xlabel('Dominant direction (°)', fontsize=24)
ax1.set_yticklabels(ax1.get_yticks(), size=24)
ax1.set_xticklabels(ax1.get_xticks(),size=24)
#plt.legend(labels=[], title = 'L1', fontsize = 24, title_fontsize = '2', loc = 'upper right')
plt.show()
