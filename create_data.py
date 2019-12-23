from plyfile import PlyData
from glob import glob
import numpy as np
import gc
from preproc import process
def ply_to_numpy(plydata):
    V = np.stack([plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']])
    V = V.transpose(1, 0)
    F = plydata['face'].data['vertex_indices']
    F = np.stack(F).astype('int32')
    return V, F



def recursive_glob(directory):
    path_subjects=[]
    for subject in sorted(glob(directory+"/*")):
        path_expressions = []
        for expression in sorted(glob(subject+"/*")):
            path_expressions.append(sorted(glob(expression+ "/*")))
        if len(path_expressions):
            path_subjects.append(path_expressions)
    return path_subjects



paths=recursive_glob('../data')

directory='../scratch_kyukon_vo'
import os
for subject in [4,6,7]:
    os.mkdir(directory+'/Subject_{0:02d}'.format(subject))
    for matrix in ['V', 'L', 'L_norm','Di', 'DiA','simple_Di']:
        os.mkdir(directory+'/Subject_{:02d}/{}'.format(subject,matrix))

for j, subject in enumerate(paths):
    if j not in [4,6,7]:
        continue
    print("subject: ",j)
    for i, expression in enumerate(paths[j]):
        list_V=[]
        list_L=[]
        list_L_norm=[]
        list_Di=[]
        list_DiA=[]
        list_simple_Di=[]

        for path in expression:
            with open(path, 'rb') as f:
                plydata = PlyData.read(f)
            V, F = ply_to_numpy(plydata)
            V, L, L_norm,Di, DiA,simple_Di=process([V, F])
            list_V.append(V)
            list_L.append(L)
            list_L_norm.append(L_norm)
            list_Di.append(Di)
            list_DiA.append(DiA)
            list_simple_Di.append(simple_Di)
        np.save(directory+'/Subject_{0:02d}/V/{1:02d}'.format(j,i),list_V)
        np.save(directory+'/Subject_{0:02d}/L/{1:02d}'.format(j,i),list_L)
        np.save(directory+'/Subject_{0:02d}/L_norm/{1:02d}'.format(j,i),list_L_norm)
        np.save(directory+'/Subject_{0:02d}/Di/{1:02d}'.format(j,i),list_Di)
        np.save(directory+'/Subject_{0:02d}/DiA/{1:02d}'.format(j,i),list_DiA)
        np.save(directory+'/Subject_{0:02d}/simple_Di/{1:02d}'.format(j,i),list_simple_Di)

        gc.collect()
