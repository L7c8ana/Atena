#!/usr/bin/env python
# coding: utf-8
# %%

# NAo classificou bem com classDA=2 (classe 1 = 48,8%), testando com classDA=1 classificou bem classe 1


# # !pip install autokeras
# # !pip install PyQt5
# # !pip install scikit-image
# # !apt-get update
# # !apt-get install ffmpeg libsm6 libxext6  -y
# # !pip install seaborn
import random, os, copy, sys
import os
import pickle
import autokeras as ak
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time
from contextlib import redirect_stdout
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, Flatten
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics

# %%
start_timeProcess = time.time()


# %%


# G = ["8", "9"]
G = ["0"]#, "6", "7"]
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


# %%
data_dirSave = "/mnt/perfil_imagem/Data/Pocos"
data_dirResults = "/mnt/perfil_imagem2/Devops2"
tipImg = ["Orig", "Log", "Gauss"]
# %%
print(strategy)


# %%

import sys
sys.path.insert(0, data_dirResults)
from data_loader import load_dataClassNewData #, load_dataClass2Borehole
# data_dirSave = "/tf/DataProjects/NewBoreholeMarcio"
CatFinal = True
AKCustom = False
ClssDA = True
ReloadPKL=False
ClassesDA = [1,2]
max_trials = 10
DataIMGChannel = 0 # 0 = Orig, 1=Log, 2=Gauss
Train = True
dataPred=False
tuner = 'hyperband' # None, greedy, bayesian', 'hyperband' 
objective="val_accuracy" # objective Imageclassifier autokeras
project_name="AKDissolution_Merge" + tipImg[DataIMGChannel] + "123"

pathmodel0 = os.path.join(data_dirResulst, "ResultTRAINWS41AKimgClassifierDissolution" + tipImg[DataIMGChannel] + "Class0", "model_autokerasWS41")
pathmodel1 = os.path.join(data_dirResulst, "ResultTRAINWS41AKimgClassifierDissolution" + tipImg[DataIMGChannel] + "Class1", "model_autokerasWS41")
pathmodel2 = os.path.join(data_dirResulst, "ResultTRAINWS41AKimgClassifierDissolution" + tipImg[DataIMGChannel] + "Class2", "model_autokerasWS41")
epochs = 100##0
ClassWeight = False
batch_size = 32 * len(G)
window_size = 41
DataModelTrain = None # [os.path.join(data_dirSave, "W1_5315-5738m_AMP.npy")]#, os.path.join(data_dirSave, "W3_5314-5883m_AMP_RAW.npy"), os.path.join(data_dirSave, "W9_AMP.npy")]
# sep = [";", ";", ","]#, ";", ",", ",", ","]#[",", ",", ";", ";", ";", ";", ",", ",", ",", ",", ","]
IDx1Borehole = ["W1", "W3", "W9"] #["W1", "W3", "W9"]#, "W6", "W8", "W10", "W11", "W12"]#["W1", "W2", "W3", "W5", "W6", "W8", "W9", "W10", "W11", "W12", "W13"]
IDx2Borehole = ["_5315-5738m_AMP", "_5314-5883m_AMP_RAW", "_AMP"] #["_5315-5738m_AMP", "_5314-5883m_AMP_RAW", "_AMP"]#, "_AMP", "_AMP", "_AMP", "_AMP", "_AMP"]#["_5315-5738m_AMP", "_5485_5915m_AMP", "_5314-5883m_AMP_RAW", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP"]
# ft = []
n_classes = 3
data_dirIMG = []
data_dirClass = []
titlefoldsave = project_name
if AKCustom:
    PathSaveFinals = os.path.join(data_dirResults, "ResultTRAINWS" + str(window_size) + titlefoldsave)
else:
    PathSaveFinals = os.path.join(data_dirResults, "ResultTRAINWS" + str(window_size) + titlefoldsave)

if not os.path.exists(PathSaveFinals):
        os.mkdir(PathSaveFinals)
if not os.path.exists(os.path.join(PathSaveFinals, "CodeUsed")):
        os.mkdir(os.path.join(PathSaveFinals, "CodeUsed"))
import shutil

shutil.copy2(os.path.realpath(__file__), os.path.join(PathSaveFinals, "Code"))
shutil.copy2(os.path.join(data_dirResults, "data_loader.py"), os.path.join(PathSaveFinals, "Code"))

if not os.path.exists(PathSaveFinals):
    os.mkdir(PathSaveFinals)

for s, n, m in zip(range(len(IDx2Borehole)),IDx1Borehole, IDx2Borehole):
    h = n + m
    data_dirIMG0 = os.path.join(data_dirSave, "Data_Amplitudes_perfil", h + ".csv")
    data_dirClass0 = os.path.join(data_dirSave, "Data_Class_perfil", n + "_FACIESRODAAN.csv")
    data_dirIMG.append(data_dirIMG0)
    data_dirClass.append(data_dirClass0)
    
data_norm = load_dataClassNewData(ReloadPKL=ReloadPKL, data_dirIMG=data_dirIMG, data_dirClass=data_dirClass, data_dirSave=data_dirSave,MakeChannels=True, dataPred=dataPred, DataTrain=DataModelTrain, LabelWell=IDx1Borehole)
# n,b = np.histogram(data_norm["FDISSOL"], bins=np.arange(0,n_classes,1))
#plt.figure()
#print(n,b)
#plt.hist(data_norm["FDISSOL"])
#plt.show()
print("Data Loaded .................")



# %%


# Nao por essa celula antes do load
import sys
sys.path.insert(0, os.path.abspath(data_dirResults))
print("Folder Actual: " + data_dirResults)
# from SlidingDataGenerator import SlidingDataGenerator
# from BoreholeImporter import Data
# from data_loader import load_dataClass, load_data
from utils import divideblocksize, ismember, cm_analysis, divideblocksizeTensor


# %%


def uniqueStr(l):
    u = list(set(l))
    i = []
    for c in l:
        i.append(int(c))
        
    return u,i


# %%


orig = data_norm['Acoustic'][:,:,0]
data_norm['Acoustic'][:,:,1] = np.nan_to_num(data_norm['Acoustic'][:,:,1], nan=0.0)
log = data_norm['Acoustic'][:,:,1]
print(log.max(), log.min())
print(orig.max(), orig.min())
# plt.figure()
# plt.imshow(orig[0:100,:])
# plt.show()
# plt.figure()
# plt.imshow(log[0:100,:])
# plt.show()
gauss = data_norm['Acoustic'][:,:,2]
print(gauss.max(), gauss.min())
print(gauss.shape)


# %%


# del Ylab
X = data_norm['Acoustic'][:,:,DataIMGChannel].reshape(orig.shape[0], orig.shape[1], 1) 
print(X.shape)
# print(data_norm['FDISSOL'])
# print(type(data_norm['FDISSOL']))
F = []
for k in data_norm['FDISSOL']:
    F.append(int(np.float32(k)))
# print(F)

Ylab = np.array(F)#data_norm['FDISSOL']
# print(Ylab)



Ylevels, Ynum = uniqueStr(Ylab)

nb_classes = n_classes

# if n_classes < 3:
#     YnumALT = np.zeros(len(Ynum))
#     for k,i in enumerate(Ynum):
#         if i == ClssAKtrain:
#             YnumALT[k] = 1
#         else:
#             YnumALT[k] = 0
    

Y = to_categorical(Ynum, num_classes=nb_classes)


__M__ = np.shape(X)[0]



train_split, test_split, validation_split = 0.6, 0.2, 0.2
LabelBH = "_".join(IDx1Borehole)
xtrain, Ytrain, xtest, Ytest, xval, Yval, ids = divideblocksizeTensor(X, Ynum, window_size, train_split, test_split, validation_split,LabelBH)
print(len(Ytrain), len(Ytest), len(Yval))

ytrain = to_categorical(Ytrain, num_classes=nb_classes)
ytest = to_categorical(Ytest, num_classes=nb_classes)
yval = to_categorical(Yval, num_classes=nb_classes)
print(len(ytrain), len(ytest), len(yval))
print(xtrain.shape, xtest.shape, xval.shape)

# create data augmentation classe 2 only data train
# Funcoes para geracao de imagens
ytrain1label = Ytrain




def h_flip(image):
    '''
    Função responsável por fazer a inversão horizontal da imagem.
    Entrada: Imagem
    Saída: Imagem invertida no sentido horizontal
    '''
    return np.fliplr(image)

def v_flip(image):
    '''
    Função responsável por fazer a inversão vertical da imagem.
    Entrada: Imagem
    Saída: Imagem invertida no sentido vertical
    '''
    return np.flipud(image)

# Dicionario para ativacao das funcoes
transformations = {'Horizontal flip': h_flip,
                   'Vertical flip': v_flip,
                  }

IDXCasos = [n for n, cl in enumerate(ytrain1label) if cl == 1 or cl == 2]
print(len(IDXCasos))

CLS = []
ccc = 0
DataNormAug = np.zeros((len(IDXCasos)*len(transformations),xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]))
for n,i in enumerate(IDXCasos):
    I = xtrain[i]
    for m,j in enumerate(transformations):
        if m == 0:
            TfI = h_flip(I)
            ttl = "Horizontal flip"
            clss = ytrain[i]
        else:
            TfI = v_flip(I)
            ttl = "Vertical flip"
            clss = ytrain[i]
        DataNormAug[ccc] = TfI
        ccc += 1
        
        CLS.append(clss)

        
# CLS = np.array(CLS)
# print(CLS.shape)
ytrainDA = np.array(CLS)#tf.keras.utils.to_categorical(CLS, num_classes=3, dtype='float32')
print(ytrainDA.shape)
        
xtrain = np.concatenate((xtrain,DataNormAug), axis=0)
ytrain = np.concatenate((ytrain,ytrainDA), axis=0)

print(xtrain.shape)
print(ytrain.shape)

def reinitialize_layer(model, initializer, layer_name, trainable=False):
    layer = model.get_layer(layer_name)
    layer.trainable = trainable
    layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
    return model



if Train:
    if os.path.exists(project_name):
        try:
            os.rmdir(project_name)
        except:
            os.remove(project_name)

    with strategy.scope():
        model0 = load_model(pathmodel0)
        with open(os.path.join(PathSaveFinals, 'model0summaryload.txt'), 'w') as f:
            with redirect_stdout(f):
                model0.summary()
        
        model1 = load_model(pathmodel1)
        with open(os.path.join(PathSaveFinals, 'model1summaryload.txt'), 'w') as f:
            with redirect_stdout(f):
                model1.summary()
                print(model1.summary())
        
        model2 = load_model(pathmodel2)
        with open(os.path.join(PathSaveFinals, 'model2summaryload.txt'), 'w') as f:
            with redirect_stdout(f):
                model2.summary()
                print(model2.summary())
                
        print(model2.summary())
        
        initializer0 = tf.keras.initializers.GlorotUniform()


        for n, i in enumerate(model0.layers):
        #     print(modelLog.layers[n].name)
            if n == 5:
                for j in model0.layers[n].layers:
                    if j.name[:19] != "batch_normalization":
                        if "kernel_initializer" in j.get_config():
                            layer = model0.layers[n].get_layer(j.name)
                            layer.trainable = True

            else:
                if "kernel_initializer" in i.get_config():
                    layer = model0.layers[n]#.get_layer(i.name)
                    layer.trainable = True

        for n, i in enumerate(model1.layers):
        #     print(modelLog.layers[n].name)
            if n == 5:
                for j in model1.layers[n].layers:
                    if j.name[:19] != "batch_normalization":
                        if "kernel_initializer" in j.get_config():
                            layer = model1.layers[n].get_layer(j.name)
                            layer.trainable = True

            else:
                if "kernel_initializer" in i.get_config():
                    layer = model1.layers[n]#.get_layer(i.name)
                    layer.trainable = True
                
        for n, i in enumerate(model2.layers):
        #     print(modelLog.layers[n].name)
            if n == 3:
                for j in model2.layers[n].layers:
                    if j.name[-3:] != "_bn":
                        if "kernel_initializer" in j.get_config():
                            layer = model2.layers[n].get_layer(j.name)
                            layer.trainable = True

            else:
                if "kernel_initializer" in i.get_config():
                    layer = model2.layers[n]#.get_layer(i.name)
                    layer.trainable = True  
        
        print(model0.optimizer.get_config())
        print(model1.optimizer.get_config())
        print(model2.optimizer.get_config())
        #{'name': 'Adam', 'learning_rate': 0.0010000000474974513, 'decay': 0.0, 'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}
        #{'name': 'Adam', 'learning_rate': 9.999999747378752e-05, 'decay': 0.0, 'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}
        #{'name': 'SGD', 'learning_rate': 0.009999999776482582, 'decay': 0.0, 'momentum': 0.0, 'nesterov': False}
        
#         
#         model0 = Model(model0.input, model0.layers[-1].output)
#         model1 = Model(model1.input, model1.layers[-1].output)
#         model2 = Model(model2.input, model2.layers[-1].output)
#         
        _input = Input(shape=(41,180,1), name="input")
        model0._name = model0._name + "0"
        model1._name = model1._name + "1"
        model2._name = model1._name + "2"
        M0 = model0(_input)
        M1 = model1(_input)
        M2 = model2(_input)
        conc = Concatenate()([M0, M1, M2])
        flat = Flatten()(conc)
        dense = Dense(3, activation="relu")(flat)
#         
        output = Dense(3, activation='softmax', name="OUT")(dense)
        auto_model = Model(_input, output)#[output1, output2, output3])
        optimizer=tf.keras.optimizers.Adam()#learning_rate=0.01)#learning_rate= 0.009999999776482582 / 10)
        auto_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        print(auto_model.summary())
       

        
        

# %%

if ClassWeight:
    weight_for_0 = (1 / np.sum(ytrain[:,0]))*(ytrain.shape[0])/3.0
    weight_for_1 = (1 / np.sum(ytrain[:,1]))*(ytrain.shape[0])/3.0
    weight_for_2 = (1 / np.sum(ytrain[:,2]))*(ytrain.shape[0])/3.0
    print(np.sum(ytrain[:,2]))

    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
    print(class_weight)

    



print(ytrain.shape)
print(yval.shape)

if Train:
#     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
    if ClassWeight:
        history = auto_model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=epochs, class_weight=class_weight)#, callbacks=[callback])
#         history = auto_model.fit(xtrain, [ytrain[:,0],ytrain[:,1],ytrain[:,2]], validation_data=(xval, [yval[:,0],yval[:,1],yval[:,2]]), epochs=epochs, class_weight=class_weight)
    else:
#         print(class_weight)
        
        history = auto_model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=epochs)#, callbacks=[callback])
        history = auto_model.fit(xtrain, ytrain, validation_data=(xval, yval), epochs=epochs)#, callbacks=[callback])
        

    print(history)

    hist = dict()
    for i in history.history.keys():
        hist.update({i: history.history[i]})
    model = auto_model.export_model()
    print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>



    with open(os.path.join(PathSaveFinals, "history.pkl"), 'wb') as f:
        pickle.dump(hist, f)


    if not os.path.exists(os.path.join(PathSaveFinals, "PlotsTrain")):
        os.mkdir(os.path.join(PathSaveFinals, "PlotsTrain"))
    plt.figure()
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))
    print("Saved plot loss in: " + os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))


    
    FOLDERmodelSv = os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size))
    try:
        model.save(os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size)), save_format="tf")
    except:
        model.save(os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size) + ".h5"))
        
    

# %%
from tensorflow.keras.models import load_model
with strategy.scope():
    model = load_model(os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size)))



model.summary()
print(model.optimizer.get_config())



# %%
if not os.path.exists(os.path.join(PathSaveFinals, "PlotsTrain")):
    os.mkdir(os.path.join(PathSaveFinals, "PlotsTrain"))

if len(G) > 1:
    predicted_y = model.predict(xtest, use_multiprocessing=True, batch_size=batch_size)
else:
    predicted_y = model.predict(xtest)

print(predicted_y.shape)
print(ytest.shape)
print(xtest.shape)

roc_data = dict()
auc = dict()
for i in range(nb_classes):
    roc_data[i] = metrics.roc_curve(ytest, predicted_y)
    auc[i] = metrics.auc(roc_data[i][0], roc_data[i][1])

f, ax = plt.subplots(figsize=[9, 6])
ax.plot(roc_data[0][0], roc_data[0][1], 'k-', label='class 0, AUC = {:4.2f}'.format(auc[0]))
ax.plot(roc_data[1][0], roc_data[1][1], 'b-', label='class 1, AUC = {:4.2f}'.format(auc[1]))
ax.plot(roc_data[2][0], roc_data[2][1], 'r-', label='class 2, AUC = {:4.2f}'.format(auc[2]))
# ax.plot([0, 1], [0, 1], 'g--')
ax.legend(loc='lower right')
f.savefig(os.path.join(PathSaveFinals, "RocClasses.png"))

APS_data = dict()
APS = dict()
for i in range(nb_classes):
    # APS_data[i] = metrics.roc_curve(y_test, preds)
    APS_data[i] = metrics.precision_recall_curve(ytest, predicted_y)
    APS[i] = metrics.average_precision_score(ytest, predicted_y)

f, ax = plt.subplots(figsize=[9, 6])
ax.plot(APS_data[0][0], APS_data[0][1], 'k-', label='class 0, AUC = {:4.2f}'.format(APS[0]))
ax.plot(APS_data[1][0], APS_data[1][1], 'b-', label='class 1, AUC = {:4.2f}'.format(APS[1]))
ax.plot(APS_data[2][0], APS_data[2][1], 'r-', label='class 2, AUC = {:4.2f}'.format(APS[2]))
ax.legend(loc='lower right')
f.savefig(os.path.join(PathSaveFinals, "Precision_recall.png"))
    
print(predicted_y)
TRUEY = np.float32(ytest)
TRUEY = np.argmax(TRUEY, axis=1)#[np.argmax(p, axis=1) for i,p in enumerate(TRUEY)]
PREDY = np.float32(predicted_y)
PREDY = np.argmax(PREDY, axis=1) #[np.argmax(p, axis=1) for i, p in enumerate(PREDY)]
print(len(TRUEY)) # list len 605
print(len(PREDY)) # list len 605
print(type(TRUEY[0])) # <class 'numpy.int64'>
print(type(PREDY[0])) # <class 'numpy.int64'>
print(TRUEY[0])
print(PREDY[0])

CMatrix = {'Test':dict()}
CMatrix['Test'] = cm_analysis(y_true=TRUEY,
                                     y_pred=PREDY,
                                     filename=os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_'),
                                     classes=Ylevels,
                                     labels=np.arange(0,3,1),
                                     figsize=(20,20),
                                     cmap='Purples') 
print("Saved plot CM in: " + os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_.png'))
print(model.optimizer.get_config())


# %%
stoptimeProcess = time.time() - start_timeProcess
# Inicia Dataframe para catálogo para salvar em json ou pkl
fileCatTrain = os.path.join(PathSaveFinals, "CatResultsTrain")
CatTrain = dict()

CatTrain.update({"Code": os.path.realpath(__file__)})
CatTrain.update({"InfoProject": {"Name": project_name, "TitleTest": titlefoldsave, "FoldSaveResults": PathSaveFinals}})
CatTrain.update({"InfoSystem": {"InfoEnviron": os.environ, "User": os.path.expanduser('~')}})
CatTrain.update({"InfoSystemTrain": {"strategy": strategy, "stringGPUs": stringGPUs, "TimeTotalProcess": stoptimeProcess}})
CatTrain.update({"Time": time.time()})
CatTrain.update({"ObjectiveCode": {"Train": Train, "dataPred": dataPred}})
CatTrain.update({"InfoModel": {"modelconfig": model.get_config(), "FolderModelsaved": FOLDERmodelSv,
                               "optimizer": model.optimizer.get_config(), "ObjImgClass": objective,
                              #"losses": losses, "metrics": metrics, 
                              "batch_size": batch_size, "history": history,
                              "window_size": window_size,
                              "AKCustom": AKCustom, "ClssDA": ClssDA, "max_trials": max_trials,
                              "tuner": tuner, "epochs": epochs, "ClassWeight": ClassWeight,
                              "DataModelTrain": DataModelTrain,
                              }})
CatTrain.update({"InfoData": {"LabelWellTrain": ["W1", "W3", "W9"], "LabelWellPred": IDx1Borehole,
                              "IDX": ids, "ClassesDA": [1,2], "TransfDA" : transformations,
                             "ReloadPKL": ReloadPKL}})

PathCat = os.path.join(PathSaveFinals, "CatDissolution")
if CatFinal == "pkl":
    with open(path + ".pkl",'wb') as f:
        pickle.dump(CatTrain,f)
if CatFinal == "json":
    result = df.to_json(orient="table")
#     parsed = json.loads(result)
#     json.dumps(parsed, indent=4) 


# %%

# %%




# %%





