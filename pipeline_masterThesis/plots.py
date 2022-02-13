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
        ax[i].set_ylabel('Density', fontsize=22)
        ax[i].set_xlabel('')
        ax[i].set_xticks(np.arange(-0.5,1.1,0.2))
        ax[i].set_xticklabels([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], size=16)
        ax[i].set_yticks(np.arange(0, 18000, step=2000))
        ax[i].set_yticklabels(np.arange(0, 18000, step=2000), size=20)
        ax[i].legend(['r','l'],title=d[i], fontsize=16, title_fontsize=20, loc = 'upper right')
    plt.tight_layout()
    plt.show()
    plt.savefig(path + 'postDistr'+name+'.png', dpi=200)
    plt.close()


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


path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/stats_VCx/'
csv_files = glob.glob(os.path.join(path, "*3p*.csv"))
for i, file in enumerate(csv_files):
    name = file[125:-7]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

Intercept = pd.melt(pd.DataFrame(np.arctan2(bpnr3p_beta2_1, bpnr3p_beta1_1)))
Intercept.insert(0,'Parameter', np.repeat('L2/3',250000))
Intercept = Intercept[['Parameter','value']]
Intercept.insert(0,'side', np.repeat('l',250000))
L4 = pd.melt(pd.DataFrame(np.arctan2(bpnr3p_beta2_1+bpnr3p_beta2_3, bpnr3p_beta1_1+bpnr3p_beta1_3)))
L4.insert(0,'Parameter', np.repeat('L4',250000))
L4 = L4[['Parameter','value']]
L4.insert(0,'side', np.repeat('l',250000))
L5 = pd.melt(pd.DataFrame(np.arctan2(bpnr3p_beta2_1+bpnr3p_beta2_4, bpnr3p_beta1_1+bpnr3p_beta1_4)))
L5.insert(0,'Parameter', np.repeat('L5',250000))
L5 = L5[['Parameter','value']]
L5.insert(0,'side', np.repeat('l',250000))
sider = pd.melt(pd.DataFrame(np.arctan2(bpnr3p_beta2_1+bpnr3p_beta2_2, bpnr3p_beta1_1+bpnr3p_beta1_2)))
sider.insert(0,'Parameter', np.repeat('L2/3',250000))
sider = sider[['Parameter','value']]
sider.insert(0,'side', np.repeat('r',250000))
L4r = pd.melt(pd.DataFrame(np.arctan2(bpnr3p_beta2_1+bpnr3p_beta2_2+bpnr3p_beta2_3, bpnr3p_beta1_1+bpnr3p_beta1_2+bpnr3p_beta1_3)))
L4r.insert(0,'Parameter', np.repeat('L4',250000))
L4r = L4r[['Parameter','value']]
L4r.insert(0,'side', np.repeat('r',250000))
L5r = pd.melt(pd.DataFrame(np.arctan2(bpnr3p_beta2_1+bpnr3p_beta2_2+bpnr3p_beta2_4, bpnr3p_beta1_1+bpnr3p_beta1_2+bpnr3p_beta1_4)))
L5r.insert(0,'Parameter', np.repeat('L5',250000))
L5r = L5r[['Parameter','value']]
L5r.insert(0,'side', np.repeat('r',250000))

l23 = pd.concat([Intercept,L4,L5,sider,L4r,L5r],ignore_index=True)


lmean = ([np.deg2rad(bpnr3p_Intercept[0].mean()),np.deg2rad(bpnr3p_layerL4[0].mean()),np.deg2rad(bpnr3p_layerL5[0].mean())])
lmedian = ([np.deg2rad(bpnr3p_Intercept[1].mean()),np.deg2rad(bpnr3p_layerL4[1].mean()),np.deg2rad(bpnr3p_layerL5[1].mean())])
lLB = ([np.deg2rad(bpnr3p_Intercept[3].mean()),np.deg2rad(bpnr3p_layerL4[3].mean()),np.deg2rad(bpnr3p_layerL5[3].mean())])
lUB = ([np.deg2rad(bpnr3p_Intercept[4].mean()),np.deg2rad(bpnr3p_layerL4[4].mean()),np.deg2rad(bpnr3p_layerL5[4].mean())])

rmean = ([np.deg2rad(bpnr3p_sider[0].mean()),np.deg2rad(bpnr3p_siderlayerL4[0].mean()),np.deg2rad(bpnr3p_siderlayerL5[0].mean())])
rmedian = ([np.deg2rad(bpnr3p_sider[1].mean()),np.deg2rad(bpnr3p_siderlayerL4[1].mean()),np.deg2rad(bpnr3p_siderlayerL5[1].mean())])
rLB = ([np.deg2rad(bpnr3p_sider[3].mean()),np.deg2rad(bpnr3p_siderlayerL4[3].mean()),np.deg2rad(bpnr3p_siderlayerL5[3].mean())])
rUB = ([np.deg2rad(bpnr3p_sider[4].mean()),np.deg2rad(bpnr3p_siderlayerL4[4].mean()),np.deg2rad(bpnr3p_siderlayerL5[4].mean())])


posteriorPlot (l23, path, '_VCx', lmean, lmedian, rmean, rmedian, lLB, lUB, rLB, rUB)


path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/y-component/'
csv_files = glob.glob(os.path.join(path, "*.csv"))
for i, file in enumerate(csv_files):
    name = file[135:-4]
    locals()[name] = pd.read_csv(file, header=None, delimiter=r"\s+")

y = []
for i in ['09', '12', '14', '17', 'VCx']:
    for j in ['SAM', 'bc', 'AS']:
        values = pd.DataFrame(pd.melt(locals()[j+'_'+i])['value'])
        values.insert(0, 'param', np.repeat(j, 250000))
        values.insert(0, 'sample', np.repeat(i, 250000))
        y.append(values)
Y = pd.DataFrame(np.vstack(y))

colors = []
fig, ax = plt.subplots(1, figsize=(15, 10))
d = ['L2/3', 'L4', 'L5']
sns.set(style="ticks")
sns.histplot(ax=ax[i],data=dat, x='value', hue = 'side', kde=True, palette=colors)
ax.set_ylabel('Density', fontsize=24)
ax.set_xlabel('')
#ax.set_xticks(np.arange(-0.5,1.1,0.2))
ax.set_xticklabels(ax.get_xticks(), size=22)
#ax.set_yticks(np.arange(0, 18000, step=2000))
ax.set_yticklabels(ax.get_yticks(), size=22)
ax.legend(['r','l'],title='Samples', fontsize=22, title_fontsize=24, loc = 'upper right')
plt.tight_layout()
plt.show()
plt.savefig(path + 'postDistr_y_09.png', dpi=200)
plt.close()


#################### graveyard: further plots ############################
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

