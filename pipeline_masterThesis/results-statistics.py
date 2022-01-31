'''
construct a csv such that all result data is in one file - ready for statistics
1. add l/r and sample id
2. average over all z-slice with same layer_id and id for tonotopic axis
3. plot the densities of both and several conditions
'''

import os
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
import seaborn as sns
import matplotlib.pyplot as plt
import glob


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
        data=pd.read_csv(os.path.join(path, 'Result_PR0'+str(i)+'_'+str(j)+'_ACx_37_Fiji_Directionality_.csv'))
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
Result_Fiji.to_csv(path+'Result_Fiji_37.csv', index=False)

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
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/'
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
                df = pd.DataFrame(z_sum)
                if df.empty:
                    continue
                else:
                    mean_circ = pd.DataFrame(z_sum).mode()[0][0]
                result.append([sampleId, side, layer_id, y, mean_circ,z_counts])
Result = pd.DataFrame(result)
for i in range(len(Result[4])):
    if Result[4][i] > np.pi:
        Result[4][i] = Result[4][i] - 2. * np.pi
Result[4] = np.rad2deg(Result[4])
Result.to_csv(path+'Result_Fiji_92-short.csv', index=False)



#### Plotting: density plots
patch_size = 37
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_37_0912141718/'
Result_long = pd.read_csv(os.path.join(path, 'Result_Fiji_'+str(patch_size)+'.csv'))
#Result_short = pd.read_csv(os.path.join(path, 'Result_Fiji_'+str(patch_size)+'-short.csv'))
dat = Result_long[Result_long['2']!='L1']
dat = dat[dat['2']!='L6']


#### density plot
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.kdeplot(data=Result_long, x=Result_long['6'], bw=0.5, color="red")
#sns.kdeplot(Result_short['4'], bw=0.5, color="red")
sns.kdeplot(data=Result_long, x=Result_long['6'], hue=Result_long['2'], fill=True, common_norm=False,
            alpha=0.4, palette="Greys_r")
#sns.kdeplot(data=Result_short, x=Result_short['4'], hue=Result_short['2'], fill=True, common_norm=False,
#            alpha=0.4, palette="Greys_r", clip = (-90.0, 90.0))
ax1.set_ylabel('Density', fontsize=24)
ax1.set_xlabel('Dominant direction (°)', fontsize=24)
ax1.set_yticklabels(ax1.get_yticks(), size=24)
ax1.set_xticklabels(ax1.get_xticks(),size=24)
plt.legend(labels=['Combined', 'L6','L5','L4','L2/3','L1'], title = '',
           fontsize = 'large', title_fontsize = '2', loc = 'upper center')
plt.show()

#### histogram + density plot
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.histplot(data=dat, x=dat['6'], hue=dat['1'], palette='colorblind', label="Combined", kde=True, bins = 50)
ax1.set_ylabel('Density', fontsize=28)
ax1.set_xlabel('Dominant direction (°)', fontsize=28)
ax1.set_yticks(y)
ax1.set_yticklabels(y, size=24)
ax1.set_xticklabels(ax1.get_xticks(),size=24)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["right","left"], fontsize = 28, loc = 'upper right')
#ax1.legend().set_visible(False)
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats_14-L2345-hist.png', dpi=200)
plt.close()


############# all same for conditions
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
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
dat = dat[(dat['0']!=18)]

dat = Result_long[(Result_long['3'].between(30,310))]
dat = dat[(dat['2']=='L2/3')]
dat = dat[(dat['4']==20)]'''


fig, (ax1) = plt.subplots(1, 1, figsize=(14, 10))
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.histplot(data=dat, x=dat['6'], hue=dat['0'], palette='colorblind', color="black", label="Combined", kde=True, bins=50)
ax1.set_ylabel('Density', fontsize=24)
ax1.set_xlabel('Dominant direction (°)', fontsize=24)
ax1.set_yticklabels(ax1.get_yticks(), size=24)
ax1.set_xticklabels(ax1.get_xticks(),size=24)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["09","12", "14", "17"], fontsize = 24, loc = 2, bbox_to_anchor=(1, 0.6))
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats_sampleID-hist-L2345.png', dpi=200)
plt.close()




############################ create further graphs for statistics ####################
# 1.) Plot small statistics over samples: mean, mode, sd
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/'
data_long = pd.read_csv(os.path.join(path, 'Result_Fiji_92.csv'))
layer_ids = ['L1', 'L2/3', 'L4', 'L5', 'L6']

#Boxplot
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.boxplot(x="0", y="6", hue="1", data = dat, palette="Greys_r", width = 0.5)
ax1.set_ylabel('Dominant direction (°)', fontsize=24)
ax1.set_xlabel('Mouse sample ID', fontsize=24)
ax1.set_xticklabels(['09', '12','14','17'],size=20)
ax1.set_yticklabels(ax1.get_yticks().astype(int), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left","right"], fontsize = 24, loc = 2, bbox_to_anchor=(1, 0.6))
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats_sampleID-boxplot-L23-y14-22.png', dpi=200)
plt.close()

#Lineplot
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.pointplot(data=dat, x='0', y='6', hue='1', ci='sd', dodge=True, markers=['o', 's'],palette='Greys_r',
                    capsize=.1)
ax1.set_ylabel('Dominant direction (°)', fontsize=24)
ax1.set_xlabel('Mouse sample ID', fontsize=24)
ax1.set_xticklabels(['09', '12','14','17'],size=20)
ax1.set_yticklabels(ax1.get_yticks().astype(int), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left","right"], fontsize = 24, loc = 2, bbox_to_anchor=(1, 0.6))
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats_sampleID-pointplot-L23-y14-22.png', dpi=200)
plt.close()


#Violinplot
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.violinplot(x="0", y="6", hue="1", data = dat, split=True, palette="colorblind", width = 0.5)
ax1.set_ylabel('Dominant direction (°)', fontsize=24)
ax1.set_xlabel('Mouse sample ID', fontsize=24)
ax1.set_xticklabels(['09', '12','14','17'],size=20)
ax1.set_yticklabels(ax1.get_yticks().astype(int), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left","right"], fontsize = 24, loc = 2, bbox_to_anchor=(1, 0.6))
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats_sampleID_L2345-violinplot.png', dpi=200)
plt.close()





########## PLot baysian results
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/stats_09/'
## plot fit
csv_files = glob.glob(os.path.join(path, "*fit*17.csv"))
for i, file in enumerate(csv_files):
    name = file[94:-7]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

f = []
for i in range(3):  #1p,2p,3p
    i+=1
    data=locals()['bpnr'+str(i)+'p_fit']
    fit = pd.DataFrame(pd.Series(data.iloc[:, 1:5].values.ravel('F')))
    names = np.concatenate((np.repeat('DIC',25), np.repeat('DIC_alt',25), np.repeat('WAIC1',25), np.repeat('WAIC2',25)))
    fit.insert(0,'criteria', names)
    fit.insert(loc=0, column='model', value=i)
    f.append(fit)
fit = pd.DataFrame(np.vstack(f))


fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.pointplot(data=fit, x=1, y=2, hue=0, ci='sd', dodge=True, markers=['o', 's', 'x'], palette='cividis_r')
ax1.set_ylabel('', fontsize=24)
ax1.set_xlabel('Model fit criteria', fontsize=24)
ax1.set_xticklabels(['DIC', 'DICalt','WAIC1','WAIC2'],size=20)
ax1.set_yticklabels(ax1.get_yticks().astype(int), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["domDir ~ side","domDir ~ side + layer", "domDir ~ side + layer + y"], fontsize = 24, loc = 'upper right')
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats17_modelFit.png', dpi=200)
plt.close()


## plot traceplots
csv_files = glob.glob(os.path.join(path, "*beta*09.csv"))
for i, file in enumerate(csv_files):
    name = file[94:-7]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

b1_1 = []
b2_1 = []
for i in range(3):  #1p,2p,3p
    i+=1
    data=locals()['bpnr'+str(i)+'p_beta1_1'].copy()
    data = pd.melt(data)
    data.insert(loc=0, column='model', value=i)
    b1_1.append(data)
    data = locals()['bpnr' + str(i) + 'p_beta2_1'].copy()
    data = pd.melt(data)
    data.insert(loc=0, column='model', value=i)
    b2_1.append(data)
b1_1 = pd.DataFrame(np.vstack(b1_1))
b2_1 = pd.DataFrame(np.vstack(b2_1))

b1_3 = []
for i in range(2):  #1p,2p,3p
    i+=2
    data=locals()['bpnr'+str(i)+'p_beta1_3'].copy()
    data = pd.melt(data)
    data.insert(loc=0, column='model', value=i)
    b1_3.append(data)
b1_3 = pd.DataFrame(np.vstack(b1_3))

b1_5 = []
data=locals()['bpnr'+str(3)+'p_beta1_5'].copy()
data = pd.melt(data)
data.insert(loc=0, column='model', value=3)
b1_5.append(data)
b1_5 = pd.DataFrame(np.vstack(b1_5))



fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
sns.lineplot(data=b1_1, x=b1_1[1], y=b1_1[2], hue=b1_1[0], ci='sd', palette='cividis_r')
ax1.set_ylabel('Beta 1: Intercept', fontsize=24)
ax1.set_xlabel('Iterations', fontsize=22)
ax1.set_xticklabels(ax1.get_xticks().astype(int),size=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
plt.legend(labels=["domDir ~ side","domDir ~ side + layer", "domDir ~ side + layer + y"], title = '',
           fontsize = 'large', title_fontsize = '2', loc = 'upper right')

plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_traceplot_b1_1.png', dpi=200)
plt.close()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
sns.lineplot(data=b2_1, x=b2_1[1], y=b2_1[2], hue=b2_1[0], ci='sd', palette='cividis_r')
ax1.set_ylabel('Beta II: Intercept', fontsize=24)
ax1.set_xlabel('Iterations', fontsize=22)
ax1.set_xticklabels(ax1.get_xticks().astype(int),size=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
plt.legend(labels=["domDir ~ side","domDir ~ side + layer", "domDir ~ side + layer + y"], title = '',
           fontsize = 'large', title_fontsize = '2', loc = 'upper right')

plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_traceplot_b2_1.png', dpi=200)
plt.close()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
sns.lineplot(data=b1_3, x=b1_3[1], y=b1_3[2], hue=b1_3[0], ci='sd', palette='cividis_r')
ax1.set_ylabel('Beta 1: L4', fontsize=24)
ax1.set_xlabel('Iterations', fontsize=22)
ax1.set_xticklabels(ax1.get_xticks().astype(int),size=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
plt.legend(labels=["domDir ~ side + layer", "domDir ~ side + layer + y"], title = '',
           fontsize = 'large', title_fontsize = '2', loc = 'upper right')

plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_traceplot_b1_3.png', dpi=200)
plt.close()


fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
sns.lineplot(data=b1_5, x=b1_5[1], y=b1_5[2], ci='sd', palette='cividis_r')
ax1.set_ylabel('Beta 1: y', fontsize=24)
ax1.set_xlabel('Iterations', fontsize=22)
ax1.set_xticklabels(ax1.get_xticks().astype(int),size=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
plt.legend(labels=["domDir ~ side + layer + y"], title = '',
           fontsize = 'large', title_fontsize = '2', loc = 'upper right')

plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_traceplot_b1_5.png', dpi=200)
plt.close()


######### plot posterior distributions
csv_files = glob.glob(os.path.join(path, "*3p_beta*09.csv"))
for i, file in enumerate(csv_files):
    name = file[94:-7]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

Intercept = pd.melt(np.arctan2(bpnr3p_beta2_1, bpnr3p_beta1_1))

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.histplot(data=Intercept, x='value')
ax1.set_ylabel('Orientations (°)', fontsize=24)
ax1.set_xlabel('Parameter values', fontsize=24)
ax1.set_xticklabels(['Mean', 'Mode','LB','UB'],size=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left, L2/3", "left, L4", "left, L5", "right, L5", "right, L4", "right, L2/3"],
           fontsize = 24, loc = 'upper right')
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_2p_params-line.png', dpi=200)
plt.close()










####plot parameters
## 1p
csv_files = glob.glob(os.path.join(path, "*1p*09.csv"))
for i, file in enumerate(csv_files):
    name = file[94:-7]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

P = []
parameters = ['Intercept', 'sider']
for i, params in enumerate(parameters):
    i+=1
    data=locals()['bpnr1p_'+params].copy()
    data = data[[0,1,3,4]]
    estm = pd.DataFrame(pd.Series(data.values.ravel('F')))
    names = np.concatenate((np.repeat('mean', 25), np.repeat('mode', 25), np.repeat('LB', 25), np.repeat('UB', 25)))
    estm.insert(0, 'criteria', names)
    estm.insert(loc=0, column='param', value=params)
    P.append(estm)
P = pd.DataFrame(np.vstack(P))

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.pointplot(data=P, x=1, y=2, hue=0, ci='sd', dodge=True, markers=['o', 's'], palette='cividis_r')
ax1.set_ylabel('Orientations (°)', fontsize=24)
ax1.set_xlabel('Parameter values', fontsize=24)
ax1.set_xticklabels(['Mean', 'Mode','LB','UB'],size=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left hemisphere", "right hemisphere"], fontsize = 24, loc = 'upper right')
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_1p_params.png', dpi=200)
plt.close()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
sns.lineplot(data=P, x=1, y=2, hue=0, ci='sd',markers=['o', 's'], palette='cividis_r')
ax1.set_ylabel('Orientations (°)', fontsize=24)
ax1.set_xlabel('Parameter values', fontsize=22)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
ax1.set_xticklabels(['Mean', 'Mode','LB','UB'],size=20)
plt.legend(labels=["left hemisphere", "right hemisphere"], title = '',
           fontsize = 'large', title_fontsize = '2', loc = 'upper right')
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_1p_params-line.png', dpi=200)
plt.close()


## 2p
csv_files = glob.glob(os.path.join(path, "*2p*09.csv"))
for i, file in enumerate(csv_files):
    name = file[94:-7]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

P2 = []
parameters = ['Intercept', 'layerL4', 'layerL5', 'siderlayerL5', 'siderlayerL4', 'sider']
for i, params in enumerate(parameters):
    i+=1
    data=locals()['bpnr2p_'+params].copy()
    data = data[[0,1,3,4]]
    estm = pd.DataFrame(pd.Series(data.values.ravel('F')))
    names = np.concatenate((np.repeat('mean', 25), np.repeat('mode', 25), np.repeat('LB', 25), np.repeat('UB', 25)))
    estm.insert(0, 'criteria', names)
    estm.insert(loc=0, column='param', value=params)
    P2.append(estm)
P2 = pd.DataFrame(np.vstack(P2))


fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.pointplot(data=P2, x=1, y=2, hue=0, ci='sd', dodge=True, palette='twilight_r')
ax1.set_ylabel('Orientations (°)', fontsize=24)
ax1.set_xlabel('Parameter values', fontsize=24)
ax1.set_xticklabels(['Mean', 'Mode','LB','UB'],size=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left, L2/3", "left, L4", "left, L5", "right, L5", "right, L4", "right, L2/3"],
           fontsize = 24, loc = 'upper right')
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_2p_params.png', dpi=200)
plt.close()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.lineplot(data=P2, x=1, y=2, hue=0, ci='sd', palette='twilight_r')
ax1.set_ylabel('Orientations (°)', fontsize=24)
ax1.set_xlabel('Parameter values', fontsize=24)
ax1.set_xticklabels(['Mean', 'Mode','LB','UB'],size=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left, L2/3", "left, L4", "left, L5", "right, L5", "right, L4", "right, L2/3"],
           fontsize = 24, loc = 'upper right')
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_2p_params-line.png', dpi=200)
plt.close()


## 3p
csv_files = glob.glob(os.path.join(path, "*3p*09.csv"))
for i, file in enumerate(csv_files):
    name = file[94:-7]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

P3 = []
parameters = ['Intercept', 'layerL4', 'layerL5', 'siderlayerL5', 'siderlayerL4', 'sider']
for i, params in enumerate(parameters):
    i+=1
    data=locals()['bpnr3p_'+params].copy()
    data = data[[0,1,3,4]]
    estm = pd.DataFrame(pd.Series(data.values.ravel('F')))
    names = np.concatenate((np.repeat('mean', 25), np.repeat('mode', 25), np.repeat('LB', 25), np.repeat('UB', 25)))
    estm.insert(0, 'criteria', names)
    estm.insert(loc=0, column='param', value=params)
    P3.append(estm)
P3 = pd.DataFrame(np.vstack(P3))


fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.pointplot(data=P3, x=1, y=2, hue=0, ci='sd', dodge=True, palette='twilight_r')
ax1.set_ylabel('Orientations (°)', fontsize=24)
ax1.set_xlabel('Parameter values', fontsize=24)
ax1.set_xticklabels(['Mean', 'Mode','LB','UB'],size=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left, L2/3", "left, L4", "left, L5", "right, L5", "right, L4", "right, L2/3"],
           fontsize = 24, loc = 'upper right')
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats09_3p_params.png', dpi=200)
plt.close()


