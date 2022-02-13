############# create plots for Bayesian statistics ##############
## 1.) posterior plots for best fit (3p)

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

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

def posteriorPlot_side(data, path, name, lmean, lmedian, rmean, rmedian,lLB, lUB, rLB, rUB):
    #fig, ax = plt.subplots(3, 1, figsize=(8, 16))
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    d = ['l', 'r']
    title = ['L', 'R']
    color = ["#5790fc", "#e42536", "#f89c20"]
    for i in range(2):
        dat = data[data['side']==d[i]]
        sns.set(style="ticks")
        sns.histplot(ax=ax[i],data=dat, x='value', hue = 'Parameter', kde=True, palette='colorblind')
        if d[i] == 'l':
            for j in range(3):
                ax[i].axvline(lmean[j], c='k', ls="-", lw=2.5)
                ax[i].axvline(lmedian[j], c='k', ls="--", lw=2.5)
                ax[i].axvline(lUB[j], c="grey", ls="--", lw=2.5)
                ax[i].axvline(lLB[j], c="grey", ls="--", lw=2.5)
        else:
            for j in range(3):
                ax[i].axvline(rmean[j], c='k', ls="-", lw=2.5)
                ax[i].axvline(rmedian[j], c='k', ls="--", lw=2.5)
                ax[i].axvline(rUB[j], c="grey", ls="--", lw=2.5)
                ax[i].axvline(rLB[j], c="grey", ls="--", lw=2.5)
        ax[i].set_ylabel("")
        ax[i].set_xlabel('Posterior distribution (°)', fontsize=34)
        ax[i].set_xticks([-0.52359878, -0.26179939,  0.        ,  0.26179939,  0.52359878,
        0.78539816,  1.04719755])
        ax[i].set_xticklabels([-30., -15.,   0.,  15.,  30.,  45.,  60.], size=30)
        ax[i].set_yticks(np.arange(0, 18000, step=2000))
        #ax[i].set_yticklabels(np.arange(0, 18000, step=2000), size=20)
        ax[i].set_yticklabels([])
        ax[i].legend(['L5','L4', 'L2/3'],title=title[i], fontsize=30, title_fontsize=34, loc = 'upper right')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'postDistrSide'+name+'.png', dpi=200)

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


path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/stats_09/'
csv_files = glob.glob(os.path.join(path, "*2p*.csv"))
for i, file in enumerate(csv_files):
    name = file[101:-7]
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


posteriorPlot (l23, path, '_14-2p', lmean, lmedian, rmean, rmedian, lLB, lUB, rLB, rUB)
posteriorPlot_side(l23, path, '_14-3p', lmean, lmedian, rmean, rmedian,lLB, lUB, rLB, rUB)


path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/y-component/'
csv_files = glob.glob(os.path.join(path, "*.csv"))
for i, file in enumerate(csv_files):
    name = file[104:-4]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

y = []
for i in ['09', '12', '14', '17', 'VCx']:
    for j in ['SAM', 'bc', 'AS']:
        values = pd.DataFrame(pd.melt(pd.DataFrame(locals()[j]))['value'])
        values.insert(0, 'param', np.repeat(j, 250000))
        #values.insert(0, 'sample', np.repeat(i, 250000))
        y.append(values)
Y = pd.DataFrame(np.vstack(y))

dat = Y[Y[0]=='09']

fig, ax = plt.subplots(1, 3, figsize=(25, 7))
d = ['L2/3', 'L4', 'L5']
sns.set(style="ticks")
sns.histplot(data=dat, x=2, hue = 1, kde=True, palette='colorblind')
#ax.axvline(lmean[i], c="blue", ls="-", lw=2.5)
#ax.axvline(lmean[i], c="blue", ls="-", lw=2.5)
#ax.axvline(lmean[i], c="blue", ls="-", lw=2.5)
ax.set_ylabel('Density', fontsize=22)
ax.set_xlabel('')
#ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticks(), size=16)
#ax.set_yticks(np.arange(0, 18000, step=2000))
ax.set_yticklabels(ax.get_yticks(), size=20)
ax.legend(['r','l'],title=d[i], fontsize=16, title_fontsize=20, loc = 'upper right')
plt.tight_layout()
plt.show()
#plt.savefig(path + 'postDistr'+name+'.png', dpi=200)
#plt.close()


#################### graveyard: further plots ############################
#### histogram + density plot
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

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.histplot(data=dat, x=dat[6], hue=dat[0], palette='colorblind', label="Combined", kde=True, bins = 90)
ax1.set_ylabel('Density', fontsize=32)
ax1.set_yticks([    0.,  5000., 10000., 15000., 20000., 25000., 30000.])
ax1.set_xlabel('Dominant direction (°)', fontsize=32)
ax1.set_yticklabels([])
ax1.set_xticklabels(ax1.get_xticks(),size=28)
handles, _ = ax1.get_legend_handles_labels()
plt.legend(handles, ["17","14", "12", "09"], fontsize = 32, loc = 'upper right')
#ax1.legend().set_visible(False)
plt.tight_layout()
plt.show()
plt.savefig(path + 'Stats-L2345-hist_20.png', dpi=200)
plt.close()

###########correct for A-P swapped -> 14,17
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/AP-020222/'
test['1'] = np.abs(test['1']-test['1'].max())

path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/stats_14/'
csv_files = glob.glob(os.path.join(path, "*1p*.csv"))
for i, file in enumerate(csv_files):
    name = file[101:-7]
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
plt.savefig(path + 'postDistr'+'_rand-1p'+'.png', dpi=200)


