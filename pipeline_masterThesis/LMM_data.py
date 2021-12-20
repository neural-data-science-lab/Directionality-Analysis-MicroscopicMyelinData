'''
construct a csv such that all result data is in one file - ready for statistics
1. add l/r and sample id
'''

import os
import numpy as np
import pandas as pd

path = 'C:/Users/Gesine/Documents/Studium/MasterCMS/MasterThesis/DataPC/'

Result_Fiji = []
sample = [12,14,17,18]
side = ['l', 'r']
for i in sample:
    for j in side:
        data=pd.read_csv(os.path.join(path, 'Result_PR0'+str(i)+'_'+str(j)+'_ACx_37_Fiji_Directionality_.csv'))
        side_id = np.array([str(j) for x in range(data.shape[0])])
        id = np.array([str(i) for x in range(data.shape[0])])
        data.insert(loc=0, column='side', value=side_id)
        data.insert(loc=0, column='id', value=id)
        Result_Fiji.append(data)
Result_Fiji = pd.DataFrame(np.vstack(Result_Fiji))
Result_Fiji.to_csv(path+'Result_Fiji.csv', index=False)

Result_OriJ = []
sample = [12,14,17,18]
side = ['l', 'r']
for i in sample:
    for j in side:
        data=pd.read_csv(os.path.join(path, 'Result_PR0'+str(i)+'_'+str(j)+'_ACx_37_OrientationJ_.csv'))
        side_id = np.array([str(j) for x in range(data.shape[0])])
        id = np.array([str(i) for x in range(data.shape[0])])
        data.insert(loc=0, column='side', value=side_id)
        data.insert(loc=0, column='id', value=id)
        Result_OriJ.append(data)
Result_OriJ = pd.DataFrame(np.vstack(Result_OriJ))
Result_OriJ.to_csv(path+'Result_OriJ_37.csv', index=False)
