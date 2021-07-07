#### Plots (later for publications) using seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_directionalityCorreted(data_l, nbr_l, data_r, nbr_r, normalize = False):
    labels = nbr_l.keys()
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=180) #, sharex=True, sharey=True
    for i, l in enumerate(labels):
        if normalize:
            title = 'Normalized directionality analysis'
            freq_l = data_l[l] / (nbr_l[l])
            freq_r = data_r[l] / (nbr_r[l])
            name = 'Directionality_norm.png'
        else:
            freq_l = data_l[l]
            freq_r = data_r[l]
            title = 'Directionality analysis'
            name = 'Directionality.png'
        ax1.plot(np.array(freq_l), np.array(data_l['Direction']), label='layer ' + l)
        ax2.plot(np.array(freq_r), np.array(data_r['Direction']), label='layer ' + l)
    ax1.invert_xaxis()
    ax1.set_ylabel('Directions in angluar degrees', fontsize=14)
    ax1.set_xlabel('Frequency', fontsize=14)
    ax1.set_title('Left')

    fig.suptitle(title, fontsize=14)
    ax1.legend(fontsize=14)
    ax2.set_yticks([])
    #ax2.set_ylabel('Directions in angluar degrees', fontsize=14)
    ax2.set_xlabel('Frequency', fontsize=14)
    ax2.set_title('Right')
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tight_layout()
    plt.show()
    # save figure
    #plt.savefig(save_path+name, dpi = 200)

path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/'
folder_left = 'Left_frangi_80/'
import_left = path + folder_left
folder_right = 'Right_frangi_40/'
import_right = path + folder_right
#corrected = pickle.load(open(import_path + "corrected.pkl", "rb"))
#nbr = eval(open(import_path + "nbr.json", "r").read())
layers = ['I', 'II/III', 'IV', 'V', 'VI']
nbr_l = pd.read_csv(import_left + 'n.csv', names = layers)
nbr_l = nbr_l.drop(nbr_l.index[[0]])
nbr_r = pd.read_csv(import_right + 'nbr_otsu.csv', names = layers)
nbr_r = nbr_r.drop(nbr_r.index[[0]])

layers.insert(0, 'Direction')
data_l = pd.read_csv(import_left + 'd.csv', names = layers)
data_l = data_l.drop(data_l.index[[0]])
data_r = pd.read_csv(import_right + 'corrected_otsu.csv', names = layers)
data_r = data_r.drop(data_r.index[[0]])

plot_directionalityCorreted(data_l, nbr_l, data_l, nbr_l, normalize = False)
