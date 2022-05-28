import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

#### 1. import files from Directionality anaylsis and set up Results table + allocate layer, etc
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
Result_Fiji.to_csv(path+'Result_Fiji_37-2.csv', index=False)


### 2. plot distributions
patch_size = 37
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_37_0912141718/'
Result_long = pd.read_csv(os.path.join(path, 'Result_Fiji_'+str(patch_size)+'.csv'))
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
sns.histplot(data=dat, x=dat['6'], hue=dat['0'], palette='colorblind', kde=True, bins = 90)
ax1.set_ylabel('Density', fontsize=28)
ax1.set_xlabel('Dominant direction (°)', fontsize=28)
#ax1.set_yticks(y)
ax1.set_yticklabels(ax1.get_yticks(), size=22)
ax1.set_xticklabels(ax1.get_xticks(),size=24)
#handles, _ = ax1.get_legend_handles_labels()
#plt.legend(handles, ["09","12","14","17"], fontsize = 28, loc = 'upper right')
#ax1.legend().set_visible(False)
ax1.legend(labels=["17","14","12","09"], title = '',
           fontsize =24, loc = 'upper center')
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats-hist-20.png', dpi=200)
plt.close()

### histograms for individuals
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.histplot(data=dat, x=dat[6], hue=dat[1], palette='colorblind', label="Combined", kde=True, bins = 90)
ax1.set_ylabel('Density', fontsize=32)
ax1.set_xlabel('Dominant direction (°)', fontsize=32)
ax1.set_yticks([   0., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.,
       9000.])
ax1.set_yticklabels([])
ax1.set_xticklabels(ax1.get_xticks(),size=28)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["right","left"], fontsize = 32, loc = 'upper right')
#ax1.legend().set_visible(False)
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats_12-L2345-hist_20.png', dpi=200)
plt.close()

### comparison between samples
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_37_0912141718/'
data_long = pd.read_csv(os.path.join(path, 'Result_Fiji_37.csv'))
layer_ids = ['L1', 'L2/3', 'L4', 'L5', 'L6']

#Boxplot
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.boxplot(x="0", y="6", hue="1", data = dat, palette="Greys_r", width = 0.5)
ax1.set_ylabel('Dominant direction (°)', fontsize=30)
ax1.set_xlabel('Mouse sample ID', fontsize=30)
ax1.set_xticklabels(['09', '12','14','17'],size=28)
ax1.set_yticklabels(ax1.get_yticks().astype(int), size=28)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left","right"], fontsize = 32, loc = 2, bbox_to_anchor=(1, 0.6))
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats_sampleID-L2345-boxplot.png', dpi=200)#
plt.close()

#Lineplot
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.pointplot(data=dat, x='0', y='6', hue='1', ci='sd', dodge=True, markers=['o', 's'],palette='Greys_r',
                    capsize=.1)
ax1.set_ylabel('Dominant direction (°)', fontsize=30)
ax1.set_xlabel('Mouse sample ID', fontsize=30)
ax1.set_xticklabels(['09', '12','14','17'],size=28)
ax1.set_yticklabels(ax1.get_yticks().astype(int), size=28)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["left","right"], fontsize = 32, loc = 2, bbox_to_anchor=(1, 0.6))
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats_sampleID-L2345-pointplot.png', dpi=200)#
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




###################################################################
### Statistics
### model fit
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/fit/'
csv_files = glob.glob(os.path.join(path, "*fit*.csv"))
for i, file in enumerate(csv_files):
    name = file[len(path):-4]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

f = []
for i in range(3):  #1p,2p,3p
    i+=1
    for j in ['09','12','14','17']:
        data=locals()['bpnr'+str(i)+'p_fit_'+j]
        fit = data[[1,3]]
        fit[1]= fit[1]-locals()['bpnr1p_fit_'+j][1].mean()
        fit[3] = fit[3] - locals()['bpnr1p_fit_'+j][3].mean()
        fit = pd.DataFrame(fit.values.ravel('F'))
        names = np.concatenate((np.repeat('DIC',25), np.repeat('WAIC',25)))
        fit.insert(0,'criteria', names)
        fit.insert(loc=0, column='model', value=i)
        fit.insert(loc=0, column='sample', value=j)
        f.append(fit)
fit = pd.DataFrame(np.vstack(f))

dat = fit[fit[2]=='DIC']

color = ["#5790fc", "#e42536", "#f89c20", "#964a8b"] # , "#656364", "k"
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks")
ax1 = sns.pointplot(data=dat, x=1, y=3, hue=0, ci='sd',  palette=color, dodge=True)
#ax1 = sns.lineplot(data=dat, x=1, y=3,hue=0, style = 0, ci='sd',  err_style="bars", palette=color, markers=True, dashes=False)
ax1.set_ylabel('DIC differences', fontsize=24)
ax1.set_xlabel('Model complexity', fontsize=24)
ax1.set_xticklabels(['domDir ~ side', 'domDir ~ side+layer','domDir ~ side+layer+y'],size=20)
ax1.set_yticklabels(ax1.get_yticks().astype(int), size=20)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["9","12", "14","17"], title='Sample', fontsize = 22, title_fontsize=24, loc = 'best')#, "Control I", "Control II"
plt.tight_layout()
plt.show()
plt.savefig(path + 'modelFit.png', dpi=200)
plt.close()

### plot posterior distributions
## for 2p and 3p
def posteriorPlot (data, path, name, lmean, lmedian, rmean, rmedian,lLB, lUB, rLB, rUB):
    #fig, ax = plt.subplots(3, 1, figsize=(8, 16))
    fig, ax = plt.subplots(1, 3, figsize=(25, 7))
    d = ['L2/3', 'L4', 'L5']
    for i in range(3):
        dat = data[data['Parameter']==d[i]]
        sns.set(style="ticks")
        sns.histplot(ax=ax[i],data=dat, x='value', hue = 'side', kde=True,palette='colorblind')
        ax[i].axvline(lmean[i], c="blue", ls="-", lw=2.5)
        ax[i].axvline(lmedian[i], c="blue", ls="--", lw=2.5)
        ax[i].axvline(lUB[i], c="grey", ls="--", lw=2.5)
        ax[i].axvline(lLB[i], c="grey", ls="--", lw=2.5)
        ax[i].axvline(rmean[i], c="orange", ls="-", lw=2.5)
        ax[i].axvline(rmedian[i], c="orange", ls="--", lw=2.5)
        ax[i].axvline(rUB[i], c="grey", ls="--", lw=2.5)
        ax[i].axvline(rLB[i], c="grey", ls="--", lw=2.5)
        ax[i].set_ylabel("")
        ax[i].set_xlabel('Posterior distribution (°)', fontsize=32)
        ax[i].set_xticks([-1.04719755, -0.78539816, -0.52359878, -0.26179939,  0.        ,
        0.26179939,  0.52359878])
        ax[i].set_xticklabels([-60., -45., -30., -15.,   0.,  15.,  30.], size=28)
        ax[i].set_yticks(np.arange(0, 18000, step=2000))
        #ax[i].set_yticklabels(np.arange(0, 18000, step=2000), size=20)
        ax[i].set_yticklabels([])
        ax[i].legend(['R','L'],title=d[i], fontsize=28, title_fontsize=32, loc = 'upper right')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'postDistr'+name+'.png', dpi=200)
    #plt.close()


### 1p
path = 'C:/Users/Gesine/Documents/ACx/'
csv_files = glob.glob(os.path.join(path, "*1p*17.csv"))
for i, file in enumerate(csv_files):
    name = file[(len(path)+7):-7]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")
I = pd.melt(pd.DataFrame(np.arctan2(beta2_1, beta1_1)))
I.insert(0,'Parameter', np.repeat('L2/3',250000))
I = I[['Parameter','value']]
I.insert(0,'side', np.repeat('l',250000))
r = pd.melt(pd.DataFrame(np.arctan2(beta2_1+beta2_2, beta1_1+beta1_2)))
r.insert(0,'Parameter', np.repeat('L2/3',250000))
r = r[['Parameter','value']]
r.insert(0,'side', np.repeat('r',250000))
l23 = pd.concat([I,r],ignore_index=True)
lmean = ([np.deg2rad(Intercept[0].mean())])
lmedian = ([np.deg2rad(Intercept[1].mean())])
lLB = ([np.deg2rad(Intercept[3].mean())])
lUB = ([np.deg2rad(Intercept[4].mean())])
rmean = ([np.deg2rad(side2[0].mean())])
rmedian = ([np.deg2rad(side2[1].mean())])
rLB = ([np.deg2rad(side2[3].mean())])
rUB = ([np.deg2rad(side2[4].mean())])

i = 0
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
d = ['L2/3', 'L4', 'L5']
dat = l23
sns.set(style="ticks")
sns.histplot(data=dat, x='value', hue = 'side', kde=True,palette='colorblind')
ax.axvline(lmean[i], c="blue", ls="-", lw=2.5)
ax.axvline(lmedian[i], c="blue", ls="--", lw=2.5)
ax.axvline(lUB[i], c="grey", ls="--", lw=2.5)
ax.axvline(lLB[i], c="grey", ls="--", lw=2.5)
ax.axvline(rmean[i], c="orange", ls="-", lw=2.5)
ax.axvline(rmedian[i], c="orange", ls="--", lw=2.5)
ax.axvline(rUB[i], c="grey", ls="--", lw=2.5)
ax.axvline(rLB[i], c="grey", ls="--", lw=2.5)
ax.set_ylabel("")
ax.set_xlabel('Posterior distribution (°)', fontsize=32)
ax.set_xticks([-0.52359878, -0.26179939,  0.        ,  0.26179939,  0.52359878,
0.78539816,  1.04719755])
ax.set_xticklabels([-30., -15.,   0.,  15.,  30.,  45.,  60.], size=28)
ax.set_yticks([    0.,  5000., 10000., 15000., 20000., 25000., 30000., 35000.])
#ax[i].set_yticklabels(np.arange(0, 18000, step=2000), size=20)
ax.set_yticklabels([])
ax.legend(['R','L'],title='Side', fontsize=28, title_fontsize=32, loc = 'upper right')
plt.tight_layout()
plt.show()
plt.savefig(path + 'postDistr'+'_17-1p'+'.png', dpi=200)


### 2/3p
path = 'C:/Users/Gesine/Documents/ACx/'
csv_files = glob.glob(os.path.join(path, "*3p*17.csv"))
for i, file in enumerate(csv_files):
    name = file[(len(path)+7):-7]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

I = pd.melt(pd.DataFrame(np.arctan2(beta2_1, beta1_1)))
I.insert(0,'Parameter', np.repeat('L2/3',250000))
I = I[['Parameter','value']]
I.insert(0,'side', np.repeat('l',250000))
L4 = pd.melt(pd.DataFrame(np.arctan2(beta2_1+beta2_3, beta1_1+beta1_3)))
L4.insert(0,'Parameter', np.repeat('L4',250000))
L4 = L4[['Parameter','value']]
L4.insert(0,'side', np.repeat('l',250000))
L5 = pd.melt(pd.DataFrame(np.arctan2(beta2_1+beta2_4, beta1_1+beta1_4)))
L5.insert(0,'Parameter', np.repeat('L5',250000))
L5 = L5[['Parameter','value']]
L5.insert(0,'side', np.repeat('l',250000))
r = pd.melt(pd.DataFrame(np.arctan2(beta2_1+beta2_2, beta1_1+beta1_2)))
r.insert(0,'Parameter', np.repeat('L2/3',250000))
r = r[['Parameter','value']]
r.insert(0,'side', np.repeat('r',250000))
L4r = pd.melt(pd.DataFrame(np.arctan2(beta2_1+beta2_2+beta2_3, beta1_1+beta1_2+beta1_3)))
L4r.insert(0,'Parameter', np.repeat('L4',250000))
L4r = L4r[['Parameter','value']]
L4r.insert(0,'side', np.repeat('r',250000))
L5r = pd.melt(pd.DataFrame(np.arctan2(beta2_1+beta2_2+beta2_4, beta1_1+beta1_2+beta1_4)))
L5r.insert(0,'Parameter', np.repeat('L5',250000))
L5r = L5r[['Parameter','value']]
L5r.insert(0,'side', np.repeat('r',250000))

l23 = pd.concat([I,L4,L5,r,L4r,L5r],ignore_index=True)


lmean = ([np.deg2rad(Intercept[0].mean()),np.deg2rad(layerL4[0].mean()),np.deg2rad(layerL5[0].mean())])
lmedian = ([np.deg2rad(Intercept[1].mean()),np.deg2rad(layerL4[1].mean()),np.deg2rad(layerL5[1].mean())])
lLB = ([np.deg2rad(Intercept[3].mean()),np.deg2rad(layerL4[3].mean()),np.deg2rad(layerL5[3].mean())])
lUB = ([np.deg2rad(Intercept[4].mean()),np.deg2rad(layerL4[4].mean()),np.deg2rad(layerL5[4].mean())])

rmean = ([np.deg2rad(sider[0].mean()),np.deg2rad(siderlayerL4[0].mean()),np.deg2rad(siderlayerL5[0].mean())])
rmedian = ([np.deg2rad(sider[1].mean()),np.deg2rad(siderlayerL4[1].mean()),np.deg2rad(siderlayerL5[1].mean())])
rLB = ([np.deg2rad(sider[3].mean()),np.deg2rad(siderlayerL4[3].mean()),np.deg2rad(siderlayerL5[3].mean())])
rUB = ([np.deg2rad(sider[4].mean()),np.deg2rad(siderlayerL4[4].mean()),np.deg2rad(siderlayerL5[4].mean())])


posteriorPlot (l23, path, '_17-3p', lmean, lmedian, rmean, rmedian, lLB, lUB, rLB, rUB)





