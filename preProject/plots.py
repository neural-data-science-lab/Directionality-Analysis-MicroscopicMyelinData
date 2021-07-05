#### Plots (later for publications) using seaborn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_directionalityCorreted(data, nbr, normalize = False):
    labels = nbr.keys()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    for i, l in enumerate(labels):
        if normalize:
            title = 'Normalized directionality analysis'
            freq = data[l] / (nbr[l])
            name = 'Directionality_norm.png'
        else:
            freq = data[l]
            title = 'Directionality analysis'
            name = 'Directionality.png'
        ax.plot(data['Direction'], freq, label='layer ' + l)
    ax.set_ylabel('Frequency of direction', fontsize=18)
    ax.set_xlabel('Directions in angluar degrees', fontsize=18)
    ax.set_title(title, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    fig.tight_layout()
    plt.show()
    # save figure
    #plt.savefig(save_path+name, dpi = 200)

path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/'
folder = 'Left_frangi_80/'
import_path = path + folder
#corrected = pickle.load(open(import_path + "corrected.pkl", "rb"))
#nbr = eval(open(import_path + "nbr.json", "r").read())
layers = ['I', 'II/III', 'IV', 'V', 'VI']
nbr = pd.read_csv(import_path + 'nbr_df.csv', names = layers)
nbr = nbr.drop(nbr.index[[0]])
layers.insert(0, 'Direction')
corrected = pd.read_csv(import_path + 'corrected_df.csv', names = layers)
corrected = corrected.drop(corrected.index[[0]])


plot_directionalityCorreted(corrected, nbr, normalize = False)

sns.set_theme()
sns.relplot(x=corrected[str(0)][:, 1], y=corrected[str(0)][:, 0], kind="line")