#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install autokeras
# !pip install PyQt5
# !pip install scikit-image
# !apt-get update
# !apt-get install ffmpeg libsm6 libxext6  -y
# !pip install seaborn
# !pip install laspy==1.5.0
# !pip install lasio
import lasio
import random, os, copy, sys
import os
import laspy
import pickle
import autokeras as ak
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.keras.models import load_model
import warnings
import seaborn as sbr
warnings.filterwarnings('ignore')


# In[2]:


# G = ["8", "9"]
G = ["0"]
stringGPUs = []
for i in G:
    GPS = "GPU:" + i
    stringGPUs.append(GPS)


print(stringGPUs)
print(",".join(G))
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(G)

if len(G) > 1:
    strategy = tf.distribute.MirroredStrategy(devices=stringGPUs)#, cross_device_ops=tf.distribute.NcclAllReduce())
else:
    strategy = tf.distribute.OneDeviceStrategy(device=stringGPUs[0])

print("Number of devices: {}".format(strategy.num_replicas_in_sync))


# In[3]:


data_dirSave = "/tf/DataProjects/NewBoreholeMarcio"
import sys
sys.path.insert(0, os.path.join(data_dirSave,"EntregaPetroDissolucao"))
from data_loader import load_dataClassNewMarcioTrain#, load_dataClass2Borehole


# In[4]:


def save_resultsPKL(results, path = None):
    with open(path,'wb') as f:
        pickle.dump(results,f)

""" function to load results """
def load_resultsPKL(path=None):
    with open(path,'rb') as f:
        results = pickle.load(f)
    return results


# In[5]:


get_ipython().system('ls /tf/DataProjects/NewBoreholeMarcio/W5_AMP.csv')


# In[6]:


import sys
sys.path.insert(0, os.path.join(data_dirSave,"EntregaPetroDissolucao"))
from data_loader import load_dataClassNewMarcioTrain#, load_dataClass2Borehole 

IDx1Borehole = "W3"
Blind = False
window_size = 41
data_dirSave = "/tf/DataProjects/NewBoreholeMarcio"
PathSaveFinals = os.path.join(data_dirSave, "ResultInferenceWS" + str(window_size) + "GaussCustomResnetWell" + IDx1Borehole)
dirPltsPred = os.path.join(PathSaveFinals, "PlotsInference")
dictResults = load_resultsPKL(path = os.path.join(dirPltsPred,'ResultsInference_' + IDx1Borehole + '.pkl'))
data_dirIMG = []
data_dirClass = []
start_time = time.time()
 
data_dirIMG.append(os.path.join(data_dirSave, dictResults["File_IMG_Well"]))
print(data_dirIMG)

if not Blind:
    data_dirClass.append(os.path.join(data_dirSave, dictResults["File_Class_Well"]))
    try:
        dpC = pd.read_csv(data_dirClass[0], delimiter=";")
    except:
        dpC = pd.read_csv(data_dirClass[0], delimiter=",")
    
    
    DepthNoInterp = dpC["Depth"]
    ClassNoInterp = dpC["faciesdissolucao"]
else:
    data_dirClass = None
data_norm = load_dataClassNewMarcioTrain(data_dirIMG=data_dirIMG, data_dirClass=data_dirClass, data_dirSave=data_dirSave,
                               MakeChannels=True, dataPred=False, DataTrain=None, LabelWell=IDx1Borehole)
print("Time load data :")   
print(time.time() - start_time)



def uniqueStr(l):
    u = list(set(l))
    i = []
    for c in l:
        i.append(int(c))

    return u,i


if Blind:
    orig = data_norm['Acoustic'][:,:,0] # image normalize
    DepthInterp = data_norm['Depth']
 


    # # Compare real and predicts 

    # In[ ]:


    cl = ['g', 'r', 'b']
    # intv = 500
    CP = np.zeros((DepthInterp.shape[0], n_classes))
    for y in range(n_classes):
        CP[:, y] = DepthInterp

    # print(dictResults["Pred_Per_Depth_Intep"])

    PredInterp = dictResults["Pred_Per_Depth_Intep"]

    PlotsVert = True
    if PlotsVert:
        inic = 4000
        fin = 5000
        fig = plt.figure(figsize=(32, 32))#constrained_layout=True)
        fig.tight_layout(pad=3.0)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax1.imshow(orig[inic:fin, :], cmap="gray")
        # ax1.set_ylim(ax1.get_ylim()[::-1])  # invertendo axes y
        # ax1.set_yticks(np.arange(fin,fin+inic,1))
        # ax1.set_ylim([DepthInterp[inic], DepthInterp[fin]])
        ax1.set_yticklabels(DepthInterp[inic:fin])
        ax1.axes.get_xaxis().set_visible(False)
        ax2.axis('off')
        CPR = CP[inic:fin,:]
        PP = PredInterp[inic:fin]
        for g, j, p in zip(np.arange(0,len(YNN),1), YNN, PP):
            ax2.plot(np.arange(0,n_classes,1), CPR[g,:], cl[int(j)])
            

        fig.savefig(os.path.join(dirPltsPred,"ShowDataREAL_PRED_ModelW1W2_" + h + str(inic) + "-" + str(fin) + ".png"))
                plt.close()

        print("Plots Saved in: " + os.path.join(dirPltsPred,"ShowDataREAL_PRED_ModelW1W2_" + h + str(inic) + "-" + str(fin) + ".png"))


  


    # !ls /usr/local/lib/python3.6/dist-packages/laspy/file.py
    # !pip install laspy==1.7.0
    # !/usr/bin/python3 -m pip install --upgrade pip
    # !pip list laspy
#     get_ipython().system('pip install laspy==1.5.0')


    # In[ ]:


    import lasio.examples
    import lasio
    from datetime import datetime
    from sys import stdout
    las = lasio.LASFile()
    las.well.DATE = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    depths = np.zeros((80))
    las.add_curve('POCO', depths, unit='str')
    las.add_curve('TOPO', depths, descr='m')
    las.add_curve('BASE', depths, descr='m')
    las.add_curve('PROPRIEDADE', depths, descr='int')
    las.add_curve('VALOR', depths, descr='int')
#     las.add_curve('CLASSIFICACAO', depths, descr='int')
    print(las.header)

    # del las.header["Well"]["Other"]

    las.write('ModeloRelatorioLAS.las', version=2)
    las = lasio.read("ModeloRelatorioLAS.las")
    print(las.sections.keys())
    dflas = las.df()
    print("####################################################################")
    print(dflas)
    # las.write(stdout)


  


else:



    # In[ ]:


    orig = data_norm['Acoustic'][:,:,0] # image normalize
    DepthInterp = data_norm['Depth']
    ClassInterp = data_norm['FDISSOL']
    # print(ClassInterp)
    Ylevels, Ynum = uniqueStr(ClassInterp)
    # print(ClassInterp[0:10])
    # print(Ynum[0:10])
    n_classes = len(Ylevels)


    # In[ ]:


    fig, ax = plt.subplots()
    sbr.set_style("darkgrid")
    df = pd.DataFrame({'labels':ClassInterp})
    ax = sbr.countplot(data=df,y='labels',saturation=0.55,edgecolor=None,linewidth=2,label=None, ax=ax, order=df['labels'].value_counts().index)
    for p in ax.patches:
        x=p.get_bbox().get_points()[1,0]
        y=p.get_bbox().get_points()[:,1]
        ax.annotate('{:.0f}'.format(x), (x+1000, y.mean()),
                ha='left', va='center') # set the alignment of the text
    plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
    ax.set_ylabel('Classes')
    fig.savefig("BarPlotNumberClass.png")


    # # Compare classifications without interpolation and interpolation 

    # In[ ]:


    cl = ['g', 'r', 'b']
    # intv = 500
    CP = np.zeros((DepthInterp.shape[0], n_classes))
    for y in range(n_classes):
        CP[:, y] = DepthInterp


    PlotsVert = True
    if PlotsVert:
        inic = 4000
        fin = 5000
        fig = plt.figure(figsize=(32, 32))#constrained_layout=True)
        fig.tight_layout(pad=3.0)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(orig[inic:fin, :], cmap="gray")
        # ax1.set_ylim(ax1.get_ylim()[::-1])  # invertendo axes y
        # ax1.set_yticks(np.arange(fin,fin+inic,1))
        # ax1.set_ylim([DepthInterp[inic], DepthInterp[fin]])
        ax1.set_yticklabels(DepthInterp[inic:fin])
        ax1.axes.get_xaxis().set_visible(False)
        ax2.axis('off')
        ax3.axis('off')
        CPR = CP[inic:fin,:]
        YNN = ClassInterp[inic:fin]
        for g, j in zip(np.arange(0,len(YNN),1), YNN):
            ax2.plot(np.arange(0,n_classes,1), CPR[g,:], cl[int(j)])
    #         ax2.set_yticklabels(CP[inic:fin,0])
        DD = np.where((np.float32(DepthNoInterp) >=  np.float32(DepthInterp[inic])) & (np.float32(DepthNoInterp) <=  np.float(DepthInterp[fin])))
        for g, l, j in zip(np.arange(0,len(DepthNoInterp[DD[0]]),1),DepthNoInterp[DD[0]], ClassNoInterp[DD[0]]):
            ax3.plot(np.arange(0,n_classes,1), [l,l,l] , cl[j])
    #         ax3.set_yticklabels(DepthNoInterp[DD[0]])
    #         ax3.set_yticklabels(CP[inic:fin,0])


    #         ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
    #                    borderaxespad=0.)  # , loc='upper left', borderaxespad=0.
    #         # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=1.)#, loc='upper left', borderaxespad=0.

    #     fig.savefig(os.path.join(dirPltsPred,"ShowDataREAL_PRED_ModelW1W2_" + h + str(inic) + "-" + str(fin) + ".png"))
    #             plt.close()

    #     print("Plots Saved in: " + os.path.join(dirPltsPred,"ShowDataREAL_PRED_ModelW1W2_" + h + str(inic) + "-" + str(fin) + ".png"))


    # # Compare real and predicts 

    # In[ ]:


    cl = ['g', 'r', 'b']
    # intv = 500
    CP = np.zeros((DepthInterp.shape[0], n_classes))
    for y in range(n_classes):
        CP[:, y] = DepthInterp

    # print(dictResults["Pred_Per_Depth_Intep"])

    PredInterp = dictResults["Pred_Per_Depth_Intep"]

    PlotsVert = True
    if PlotsVert:
        inic = 4000
        fin = 5000
        fig = plt.figure(figsize=(32, 32))#constrained_layout=True)
        fig.tight_layout(pad=3.0)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(orig[inic:fin, :], cmap="gray")
        # ax1.set_ylim(ax1.get_ylim()[::-1])  # invertendo axes y
        # ax1.set_yticks(np.arange(fin,fin+inic,1))
        # ax1.set_ylim([DepthInterp[inic], DepthInterp[fin]])
        ax1.set_yticklabels(DepthInterp[inic:fin])
        ax1.axes.get_xaxis().set_visible(False)
        ax2.axis('off')
        ax3.axis('off')
        CPR = CP[inic:fin,:]
        YNN = ClassInterp[inic:fin]
        PP = PredInterp[inic:fin]
        for g, j, p in zip(np.arange(0,len(YNN),1), YNN, PP):
            ax2.plot(np.arange(0,n_classes,1), CPR[g,:], cl[int(j)])
            if int(p) > 0:
                ax3.plot(np.arange(0,n_classes,1), CPR[g,:], cl[int(p)-1])

    #     fig.savefig(os.path.join(dirPltsPred,"ShowDataREAL_PRED_ModelW1W2_" + h + str(inic) + "-" + str(fin) + ".png"))
    #             plt.close()

    #     print("Plots Saved in: " + os.path.join(dirPltsPred,"ShowDataREAL_PRED_ModelW1W2_" + h + str(inic) + "-" + str(fin) + ".png"))


    # In[ ]:


    # !ls /usr/local/lib/python3.6/dist-packages/laspy/file.py
    # !pip install laspy==1.7.0
    # !/usr/bin/python3 -m pip install --upgrade pip
    # !pip list laspy
    get_ipython().system('pip install laspy==1.5.0')


    # In[ ]:


    import lasio.examples
    import lasio
    from datetime import datetime
    from sys import stdout
    las = lasio.LASFile()
    las.well.DATE = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    depths = np.zeros((80))
    las.add_curve('POCO', depths, unit='str')
    las.add_curve('TOPO', depths, descr='m')
    las.add_curve('BASE', depths, descr='m')
    las.add_curve('PROPRIEDADE', depths, descr='int')
    las.add_curve('VALOR', depths, descr='int')
    las.add_curve('CLASSIFICACAO', depths, descr='int')
    print(las.header)

    # del las.header["Well"]["Other"]

    las.write('ModeloRelatorioLAS.las', version=2)
    las = lasio.read("ModeloRelatorioLAS.las")
    print(las.sections.keys())
    dflas = las.df()
    print("####################################################################")
    print(dflas)
    # las.write(stdout)


    # In[ ]:




