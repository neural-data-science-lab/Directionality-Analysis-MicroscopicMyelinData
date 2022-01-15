import os
import numpy as np
import pandas as pd
from scipy.stats import circmean
import seaborn as sns
import matplotlib.pyplot as plt

path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_0912141718/'
data_short = pd.read_csv(os.path.join(path, 'Result_Fiji_92_mode-short.csv'))
Result_long = pd.read_csv(os.path.join(path, 'Result_Fiji_92.csv'))

def Convert(lst):
    return [ -i for i in lst ]

data_long.loc[data_long['1'] == 'l', ['6']] = Convert(data_long[data_long['1'] == 'l']['6'])
data_long.to_csv(path+'Result_Fiji_92-leftInvert.csv', index=False)



#### Plotting: density plots
patch_size = 92
path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/Result_92_12141718/'
Result_long = pd.read_csv(os.path.join(path, 'Result_Fiji_'+str(patch_size)+'.csv'))
Result_short = pd.read_csv(os.path.join(path, 'Result_Fiji_'+str(patch_size)+'-short.csv'))
Result_mode = pd.read_csv(os.path.join(path, 'Result_Fiji_'+str(patch_size)+'_mode-short.csv'))

#### density plot
dat = Result_long[(Result_long['2']!='L1')]
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
#sns.set(font_scale = 1.5)
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.kdeplot(data=dat, x=dat['6'], bw=0.5, color="red")
#sns.kdeplot(Result_short['4'], bw=0.5, color="red")
sns.kdeplot(data=dat, x=dat['6'], hue=dat['0'], fill=True, common_norm=False,
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
dat = Result_long[(Result_long['2']!='L1')]
dat = dat[dat['0']!=9]
dat = dat[dat['1']=='r']
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
#sns.set(font_scale = 1.5)
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.histplot(data=dat, x=dat['6'], color="black", kde=True, bins=32)#label="Combined",
ax1.set_ylabel('Density', fontsize=20)
ax1.set_xlabel('Dominant direction (°)', fontsize=20)
ax1.set_yticklabels(ax1.get_yticks(), size=20)
ax1.set_xticklabels(ax1.get_xticks(),size=20)
plt.legend(labels=['right: w/o L1'], title = '', fontsize = 20, title_fontsize = '2', loc = 'upper right')
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

fig, (ax1) = plt.subplots(1, 1, figsize=(14, 10))
sns.set(style="ticks")
plt.xlim(-90, 90)
sns.histplot(data=dat, x=dat['4'], color="black",  kde=True, bins=50)#label="Combined",
ax1.set_ylabel('Density', fontsize=24)
ax1.set_xlabel('Dominant direction (°)', fontsize=24)
ax1.set_yticklabels(ax1.get_yticks(), size=24)
ax1.set_xticklabels(ax1.get_xticks(),size=24)
#plt.legend(labels=[], title = 'L1', fontsize = 24, title_fontsize = '2', loc = 'upper right')
plt.show()
