#!/usr/bin/env python
# coding: utf-8
# %%

# Classificou bem classe 2 com 100 %


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
import tensorflow
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, Lambda #, MaxPooling2D, Conv2D, Flatten, Concatenate, Add, BatchNormalization
from tensorflow.keras.layers import Activation, Layer
from tensorflow.keras.callbacks import LearningRateScheduler
# snippet of using the ReduceLROnPlateau callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics

# %%


#get_ipython().system('ls -l /tf/DataProjects/NewBoreholeMarcio/')


# %%
Train = False
PredFinalModel = False
# G = ["8", "9"]
G = ["0", "1"]
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


# # !mv /tf/DataProjects/NewBoreholeMarcio/dataNormW_0.pkl /tf/DataProjects/NewBoreholeMarcio/dataNormW1_W3_W9.pkl


# %%


data_dirSave = "/tf/DataProjects/NewBoreholeMarcio"
import sys
sys.path.insert(0, os.path.join(data_dirSave,"EntregaPetroDissolucao"))
from data_loader import load_dataClassNewData#Train#, load_dataClass2Borehole
# data_dirSave = "/tf/DataProjects/NewBoreholeMarcio"

AKCustom = False
max_trials = 5
epochs = 100
ClssDA = False
ClassesDA = [1,2]
batch_size = 32
window_size = 41
DataModel = None # [os.path.join(data_dirSave, "W1_5315-5738m_AMP.npy")]#, os.path.join(data_dirSave, "W3_5314-5883m_AMP_RAW.npy"), os.path.join(data_dirSave, "W9_AMP.npy")]
# sep = [";", ";", ","]#, ";", ",", ",", ","]#[",", ",", ";", ";", ";", ";", ",", ",", ",", ",", ","]
IDx1Borehole = ["W1", "W3", "W9"] #["W1", "W3", "W9"]#, "W6", "W8", "W10", "W11", "W12"]#["W1", "W2", "W3", "W5", "W6", "W8", "W9", "W10", "W11", "W12", "W13"]
IDx2Borehole = ["_5315-5738m_AMP", "_5314-5883m_AMP_RAW", "_AMP"] #["_5315-5738m_AMP", "_5314-5883m_AMP_RAW", "_AMP"]#, "_AMP", "_AMP", "_AMP", "_AMP", "_AMP"]#["_5315-5738m_AMP", "_5485_5915m_AMP", "_5314-5883m_AMP_RAW", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP"]
# ft = []
ReloadIDX = False
LabelBH = "".join(IDx1Borehole)
n_classes = 3
data_dirIMG = []
data_dirClass = []
titlefoldsave = "Train2BestModelsChannelsGAUSSandOrig"
if AKCustom:
    PathSaveFinals = os.path.join(data_dirSave, "ResultTRAINWS" + str(window_size) + titlefoldsave)
else:
    PathSaveFinals = os.path.join(data_dirSave, "ResultTRAINWS" + str(window_size) + titlefoldsave)
    
# /home/luciana/NewBoreholeMarcio/ResultTRAINWS41AKimgClassifierDissolutionGAUSS
# foldermodel1 = os.path.join(data_dirSave, "ResultTRAINWS" + str(window_size) + "TestAKModelGauss")
foldermodelLog = os.path.join(data_dirSave, "ResultTRAINWS" + str(window_size) + "AKimgClassifierDissolutionGAUSS")
foldermodelOrig = os.path.join(data_dirSave, "ResultTRAINWS" + str(window_size) + "AKimgClassifierDissolutionOrig")    

if not os.path.exists(PathSaveFinals):
    os.mkdir(PathSaveFinals)

for s, n, m in zip(range(len(IDx2Borehole)),IDx1Borehole, IDx2Borehole):
    h = n + m
    data_dirIMG0 = os.path.join(data_dirSave, h + ".csv")
    data_dirClass0 = os.path.join(data_dirSave, n + "_FACIESRODAAN.csv")
    data_dirIMG.append(data_dirIMG0)
    data_dirClass.append(data_dirClass0)
    
#load_dataClassNewDat
data_norm = load_dataClassNewData(data_dirIMG=data_dirIMG, data_dirClass=data_dirClass, data_dirSave=data_dirSave,
                               MakeChannels=True, dataPred=False, DataTrain=DataModel, LabelWell=IDx1Borehole)
n,b = np.histogram(data_norm["FDISSOL"], bins=[0,1,2,3])
plt.figure()
print(n,b)
plt.hist(data_norm["FDISSOL"])
#plt.show()


# %%


# Nao por essa celula antes do load
import sys
#sys.path.insert(0, os.path.abspath(os.path.join(data_dirSave,'EntregaPetroDissolucao','BoreholeTools')))
print("Folder Actual: " + os.getcwd())
#from SlidingDataGenerator import SlidingDataGenerator
#from BoreholeImporter import Data
# from data_loader import load_dataClass, load_data
from utils import ismember, cm_analysis, divideblocksizeTensor


# %%


def uniqueStr(l):
    u = list(set(l))
    i = []
    for c in l:
        i.append(int(c))
        
    return u,i


# %%


orig = data_norm['Acoustic'][:,:,0].reshape(data_norm['Acoustic'].shape[0], data_norm['Acoustic'].shape[1], 1) 
data_norm['Acoustic'][:,:,1] = np.nan_to_num(data_norm['Acoustic'][:,:,1], nan=0.0)
log = data_norm['Acoustic'][:,:,1].reshape(orig.shape[0], orig.shape[1], 1) 
gauss = data_norm['Acoustic'][:,:,2].reshape(orig.shape[0], orig.shape[1], 1)
print(log.max(), log.min())
print(orig.max(), orig.min())
# plt.figure()
# plt.imshow(orig[0:100,:])
# plt.show()
# plt.figure()
# plt.imshow(log[0:100,:])
# plt.show()
# gauss = data_norm['Acoustic'][:,:,2].reshape(orig.shape[0], orig.shape[1], 1) 
# print(gauss.shape)


# %%


# del Ylab
X = data_norm['Acoustic'] #[:,:,1]# orig.reshape(orig.shape[0], orig.shape[1], 1) 
print(X.shape)
# print(data_norm['FDISSOL'])
# print(type(data_norm['FDISSOL']))
#Ylab = data_norm['FDISSOL']
F = []
for k in data_norm['FDISSOL']:
    F.append(int(np.float32(k)))
    #print(F)

Ylab = np.array(F)#data_norm['FDISSOL']
print(Ylab)

# %%

Ylevels, Ynum = uniqueStr(Ylab)

nb_classes = len(Ylevels)

__M__ = np.shape(X)[0]




train_split, test_split, validation_split = 0.6, 0.2, 0.2
xtrain_Orig, Ytrain, xtest_Orig, Ytest, xval_Orig, Yval, ids = divideblocksizeTensor(gauss, Ynum, window_size, train_split, test_split, validation_split, LabelBH, reloadIDX=ReloadIDX)
xtrain_Log, Ytrain, xtest_Log, Ytest, xval_Log, Yval, ids = divideblocksizeTensor(log, Ynum, window_size, train_split, test_split, validation_split, LabelBH, reloadIDX=ReloadIDX)
# xtrain_Gauss, Ytrain, xtest_Gauss, Ytest, xval_Gauss, Yval, ids = divideblocksizeTensor(gauss, Ynum, window_size, train_split, test_split, validation_split)
       
print(xtrain_Orig.shape, xtest_Orig.shape, xval_Orig.shape)
print(xtrain_Log.shape, xtest_Log.shape, xval_Log.shape)
# print(xtrain_Gauss.shape, xtest_Gauss.shape, xval_Gauss.shape)
print(len(Ytrain), len(Ytest), len(Yval))


ytrain = to_categorical(Ytrain, num_classes=nb_classes)
ytest = to_categorical(Ytest, num_classes=nb_classes)
yval = to_categorical(Yval, num_classes=nb_classes)

# create data augmentation classe 2 only data train
# Funcoes para geracao de imagens
ytrain1label = Ytrain

######################################################



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

# %%


if Train:
    # Dicionario para ativacao das funcoes
    transformations = {'Horizontal flip': h_flip,
                       'Vertical flip': v_flip,
                      }
    #############################################################################################
    IDXCasos = [n for n, cl in enumerate(ytrain1label) if cl == 1 or cl == 2]
    print(len(IDXCasos))

    CLS = []
    ccc = 0
    DataNormAugOrig = np.zeros((len(IDXCasos)*len(transformations),xtrain_Orig.shape[1], xtrain_Orig.shape[2], xtrain_Orig.shape[3]))
    for n,i in enumerate(IDXCasos):
        I = xtrain_Orig[i]
        for m,j in enumerate(transformations):
            if m == 0:
                TfI = h_flip(I)
                ttl = "Horizontal flip"
                clss = ytrain[i]
            else:
                TfI = v_flip(I)
                ttl = "Vertical flip"
                clss = ytrain[i]
            DataNormAugOrig[ccc] = TfI
            ccc += 1

            CLS.append(clss)

    # CLS = np.array(CLS)
    # print(CLS.shape)
    ytrainDAOrig = np.array(CLS)#tf.keras.utils.to_categorical(CLS, num_classes=3, dtype='float32')
    print(ytrainDAOrig.shape)
    xtrainOrig = np.concatenate((xtrain_Orig,DataNormAugOrig), axis=0)# classifica bem a classe 2
    ytrainOrig = np.concatenate((ytrain,ytrainDAOrig), axis=0)# classifica bem a classe 2
    ################################################################################################
    IDXCasos = [n for n, cl in enumerate(ytrain1label) if cl == 1 or cl == 2]
    print(len(IDXCasos))
    CLS = []
    ccc = 0
    DataNormAugLog = np.zeros((len(IDXCasos)*len(transformations),xtrain_Log.shape[1], xtrain_Log.shape[2], xtrain_Log.shape[3]))
    for n,i in enumerate(IDXCasos):
        I = xtrain_Log[i]
        for m,j in enumerate(transformations):
            if m == 0:
                TfI = h_flip(I)
                ttl = "Horizontal flip"
                clss = ytrain[i]
            else:
                TfI = v_flip(I)
                ttl = "Vertical flip"
                clss = ytrain[i]
            DataNormAugLog[ccc] = TfI
            ccc += 1

            CLS.append(clss)


    # CLS = np.array(CLS)
    # print(CLS.shape)
    ytrainDALog = np.array(CLS)#tf.keras.utils.to_categorical(CLS, num_classes=3, dtype='float32')
    print(ytrainDALog.shape)
    xtrainLog = np.concatenate((xtrain_Log,DataNormAugLog), axis=0) # classifica bem a classe 1
    ytrainLog = np.concatenate((ytrain,ytrainDALog), axis=0) # classifica bem a classe 1
    #########################################################################################################


    def reinitialize_layer(model, initializer, layer_name, trainable=False):
        layer = model.get_layer(layer_name)
        layer.trainable = trainable
        layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
        return model



    with strategy.scope():

    #     model1 = load_model(os.path.join(foldermodel1, "model_autokerasWS" + str(window_size)))# gauss
        modelLog = load_model(os.path.join(foldermodelLog, "model_autokerasWS" + str(window_size)))#log 
        modelOrig = load_model(os.path.join(foldermodelOrig, "model_autokerasWS" + str(window_size)))#orig
    
        modelLog.summary()
        #modelOrig.summary()
        print(modelLog.optimizer.get_config())
        print(modelOrig.optimizer.get_config())
        
        from contextlib import redirect_stdout

        with open(os.path.join(PathSaveFinals, 'modelLOGsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                modelLog.summary()
            
        with open(os.path.join(PathSaveFinals, 'modelOrigsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                modelOrig.summary()
                
        
        
        print(modelLog.layers[-1].get_config())
        print(modelOrig.layers[-1].get_config())
        
        

        # print(model3.layers[-1].get_config())
        # print(model1.optimizer.get_config())
        print(modelLog.optimizer.get_config()) # {'name': 'Adam', 'learning_rate': 9.999999747378752e-06, 'decay': 0.0, 'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}
        print(modelOrig.optimizer.get_config()) # {'name': 'SGD', 'learning_rate': 0.009999999776482582, 'decay': 0.0, 'momentum': 0.0, 'nesterov': False}
        print(modelLog.get_config())
        print(modelOrig.get_config())
        
        # model2.summary()
        # model3.summary()

        initializer0 = tensorflow.keras.initializers.GlorotUniform()


        for n, i in enumerate(modelLog.layers):
        #     print(modelLog.layers[n].name)
            if n == 4:
                for j in modelLog.layers[n].layers:
        #             print(j.name)
                    if j.name[:19] != "batch_normalization":#or j.name[-12:] == "ormalization":
                        print(j.name[:19])
        #                 print(j.name[-12:])

                        if "kernel_initializer" in j.get_config():
                            initializer = j.kernel_initializer
                            reinitialize_layer(modelLog.layers[n], initializer, j.name, trainable=True)
                        else:
                            reinitialize_layer(modelLog.layers[n], initializer0, j.name, trainable=True)


            else:
                if "kernel_initializer" in modelLog.layers[n].get_config():
                    initializer = modelLog.layers[n].kernel_initializer
                    reinitialize_layer(modelLog, initializer, modelLog.layers[n].name, trainable=True)
                else:
                    reinitialize_layer(modelLog, initializer0, modelLog.layers[n].name, trainable=True)

        for n, i in enumerate(modelOrig.layers):
            print("LAYERS MODEL ORIG >>>>>>>>>>>>>>")
            print(modelOrig.layers[n].name)
            if n == 4:
                for j in modelOrig.layers[n].layers:
        #             print(j.name)
                    if j.name[-3:] == "_bn":#or j.name[-12:] == "ormalization":
        #                 print(j.name[-3:])
        #                 print(j.name[-12:])

                        if "kernel_initializer" in j.get_config():
                            initializer = j.kernel_initializer
                            reinitialize_layer(modelOrig.layers[n], initializer, j.name, trainable=True)
                        else:
                            reinitialize_layer(modelOrig.layers[n], initializer0, j.name, trainable=True)


            else:
                if "kernel_initializer" in modelOrig.layers[n].get_config():
                    initializer = modelOrig.layers[n].kernel_initializer
                    reinitialize_layer(modelOrig, initializer, modelOrig.layers[n].name, trainable=True)
                else:
                    reinitialize_layer(modelOrig, initializer0, modelOrig.layers[n].name, trainable=True)


        ######################

        modelOrig2 = tensorflow.keras.models.clone_model(modelOrig)
        modelLog._name = modelLog.name + "_Log"
        modelOrig._name = modelOrig.name + "_Orig"
        modelOrig2._name = modelOrig.name + "_Orig2"
        # 
        # inpSeparate1 = Lambda(lambda x: x[0])(input_)
        # model1_ = model1(inpSeparate1)
        # inpSeparate2 = Lambda(lambda x: x[1])(input_)
        # model2_ = model2(inpSeparate2)
        # inpSeparate3 = Lambda(lambda x: x[2])(input_)
        # model3_ = model3(inpSeparate3)
        # for n, layer in enumerate(model1.layers):
        #     layer._name = layer.name + str("_1")

        for n, layer in enumerate(modelLog.layers):
            layer._name = layer.name + str("_Log")

        for n, layer in enumerate(modelOrig.layers):
            layer._name = layer.name + str("_Orig")

        for n, layer in enumerate(modelOrig2.layers):
            layer._name = layer.name + str("_Orig2")
        # 
        outputs = ["0", "1", "2"]
        conc_out = Concatenate()([modelOrig.layers[-1].output, modelLog.layers[-1].output])
        output = Dense(3, activation="sigmoid", name="012")(conc_out)
        
#         output0 = Dense(1, activation='sigmoid', name="0")(modelOrig.layers[-1].output)
#         output1 = Dense(1, activation='sigmoid', name="1")(modelLog.layers[-1].output)
#         output2 = Dense(1, activation='sigmoid', name="2")(modelOrig.layers[-1].output)
#         conc_out = Concatenate()([output0, output1, output2])
#         output = Dense(3, activation="softmax", name="012")(conc_out)
        model = Model([modelOrig.input, modelLog.input], output)#[output0, output1, output2])
#         model = Model([modelOrig.input, modelLog.input],[output0, output1, output2])


        losses = {xn: "categorical_crossentropy" for n, xn in enumerate(outputs)}

        metrics = {xn: "accuracy" for n, xn in enumerate(outputs)}

        lossWeights = {xn: 1.0 for n, xn in enumerate(outputs)}
        INIT_LR =  0.009999999776482582 * 100
#         INIT_LR = 9.999999747378752e-06
#         {'name': 'Adam', 'learning_rate': 9.999999747378752e-06, 'decay': 0.0, 'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}
#         print(modelOrig.optimizer.get_config()) # {'name': 'SGD', 'learning_rate': 0.009999999776482582, 'decay': 0.0, 'momentum': 0.0, 'nesterov': False}
        optimizer=tf.keras.optimizers.Adam()#learning_rate= INIT_LR)#, decay=0.0, momentum=0.0, nesterov=False)#(learning_rate=INIT_LR)#, decay=INIT_LR / self.epochs)#, epsilon=1e-05)
#         optimizer=tf.keras.optimizers.Adam(learning_rate= 1e-05 * 100)#, decay=INIT_LR/epochs, beta_1=0.8999999761581421, beta_2=0.9990000128746033, epsilon=1e-07, amsgrad=False)#(learning_rate=INIT_LR)#, decay=INIT_LR / self.epochs)#, epsilon=1e-05)


        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
#         model.compile(loss=losses, optimizer=optimizer, metrics=metrics)

        dictCat = dict()        
        import datetime

        dictCat.update({"DateTime": datetime.datetime.now()})
        dictCat.update({"window_size": window_size})
        dictCat.update({"modelconfig": model.get_config()})
        dictCat.update({"optimizer": model.optimizer.get_config()})
        dictCat.update({"losses": losses})
        dictCat.update({"metrics": losses})
        
        config = model.to_json()
#         loaded_model = tensorflow.keras.models.model_from_json(config)
        # model.compile(loss="mse", optimizer=optimizer, metrics=["mse"]) #loss=losses, loss_weights=lossWeights, optimizer=tf.keras.optimizers.Adam(),#learning_rate=INIT_LR),#, decay=INIT_LR / self.epochs, epsilon=1e-05),
                      # metrics=metrics) #epsilon=1e-07
#         print(model.summary())
#         print("LR:")        
#         print(modelLog.optimizer.learning_rate)
#         print(modelOrig.optimizer.learning_rate)
#         print(xtrainOrig.shape)
#         print(ytrainOrig.shape)
#         print(xval_Orig.shape)
#         print(yval.shape)

            
    print(xtrainOrig.shape)
    print(xtrainLog.shape)
    print(ytrainOrig.shape)
    print(ytrainLog.shape)
    
    weight_for_0 = (1 / np.sum(ytrainOrig[:,0]))*(ytrainOrig.shape[0])/3.0
    weight_for_1 = (1 / np.sum(ytrainOrig[:,1]))*(ytrainOrig.shape[0])/3.0
    weight_for_2 = (1 / np.sum(ytrainOrig[:,2]))*(ytrainOrig.shape[0])/3.0

    #class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
#    class_weight = {0: {0: weight_for_0}, 1: {0: weight_for_1}, 2: {0:weight_for_2}}
    # print(classif ClassWeight:
     
                    
    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
    print(class_weight)

    # { 'output1': {0: ratio_1 , 1: ratio_2} , 'output2': {0: ratio_3 , 1: ratio_4}}
    model_name = os.path.join(PathSaveFinals,"best_model")# "weights_{epoch:03d}_{loss:.4f}_{val_loss:.4f}.hd5")
#     if os.path.isfile(model_name):
#         print("model_name " + "exist")
#         os.remove(model_name)
#         print("model_name " + "removed")
    
    callback = ModelCheckpoint(model_name, monitor="val_accuracy", verbose=1, save_best_only=True, mode='max')
#     history = model.fit([xtrainOrig[0:100,:,:,0], xtrainLog[0:100,:,:,0]], ytrainOrig[0:100,:], validation_data=([xval_Orig[0:100,:,:,0], xval_Log[0:100,:,:,0]], yval[0:100,:]), epochs=epochs, shuffle=True, class_weight=class_weight, callbacks=[callback])
    history = model.fit([xtrainOrig, xtrainLog], ytrainOrig, validation_data=([xval_Orig, xval_Log], yval), epochs=epochs, shuffle=True, class_weight=class_weight, callbacks=[callback])


#     history = model.fit([xtrainOrig, xtrainLog], [ytrainOrig[:,0], ytrainOrig[:,1], ytrainOrig[:,2]], validation_data=([xval_Orig, xval_Log,xval_Orig], [yval[:,0], yval[:,1], yval[:,2]]), epochs=epochs, shuffle=True, class_weight=class_weight)
#     model = model.export_model()
#     print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>


    
    model.save(os.path.join(PathSaveFinals,"modelFINAL_autokerasWS" + str(window_size)), save_format="tf")
    
#     model.save(os.path.join(PathSaveFinals,"modelFINAL_H5_autokerasWS" + str(window_size) + ".h5"))
        
    # save history
    hist = dict()
    for i in history.history.keys():
        hist.update({i: history.history[i]})

    with open(os.path.join(PathSaveFinals, "history.pkl"), 'wb') as f:
        pickle.dump(hist, f)
    
    if not os.path.exists(os.path.join(PathSaveFinals, "PlotsTrain")):
        os.mkdir(os.path.join(PathSaveFinals, "PlotsTrain"))
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))
    print("Saved plot loss in: " + os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))
    
    from sklearn import metrics

    if len(G) > 1:
        predicted_y = model.predict([xtest_Orig, xtest_Log])#, use_multiprocessing=True, batch_size=batch_size)
    else:
        predicted_y = model.predict([xtest_Orig, xtest_Log])

    print(predicted_y)
    print(predicted_y[0].shape)
    print(len(predicted_y))
    print(ytest.shape)

    roc_data = dict()
    auc = dict()
    for i in range(nb_classes):
        roc_data[i] = metrics.roc_curve(ytest[:, i], predicted_y[i])
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
        APS_data[i] = metrics.precision_recall_curve(ytest[:, i], predicted_y[i])
        APS[i] = metrics.average_precision_score(ytest[:, i], predicted_y[i])

    f, ax = plt.subplots(figsize=[9, 6])
    ax.plot(APS_data[0][0], APS_data[0][1], 'k-', label='class 0, AUC = {:4.2f}'.format(APS[0]))
    ax.plot(APS_data[1][0], APS_data[1][1], 'b-', label='class 1, AUC = {:4.2f}'.format(APS[1]))
    ax.plot(APS_data[2][0], APS_data[2][1], 'r-', label='class 2, AUC = {:4.2f}'.format(APS[2]))
    ax.legend(loc='lower right')
    f.savefig(os.path.join(PathSaveFinals, "Precision_recall.png"))

    TRUEY = np.float32(ytest)
    print(TRUEY.shape)
    TRUEY = np.argmax(TRUEY, axis=1)#[np.argmax(p, axis=1) for i,p in enumerate(TRUEY)]
    print(TRUEY.shape)

    PREDY = np.float32(predicted_y)
    print(PREDY.reshape((PREDY.shape[1], PREDY.shape[0])).shape)
    PREDY = np.argmax(PREDY.reshape((PREDY.shape[1], PREDY.shape[0])), axis=1) #[np.argmax(p, axis=1) for i, p in enumerate(PREDY)]
    print(PREDY.shape)
    # print(len(TRUEY)) # list len 605
    # print(len(PREDY)) # list len 605
    # print(type(TRUEY[0])) # <class 'numpy.int64'>
    # print(type(PREDY[0])) # <class 'numpy.int64'>
    # print(TRUEY[0])
    # print(PREDY[0])

    CMatrix = {'Test':dict()}
    CMatrix['Test'] = cm_analysis(y_true=TRUEY,
                                         y_pred=PREDY,
                                         filename=os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_'),
                                         classes=Ylevels,
                                         labels=np.arange(0,3,1),
                                         figsize=(20,20),
                                         cmap='Purples') 
    print("Saved plot CM in: " + os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_.png'))
    print("Saved plot loss in: " + os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))
    
    if not os.path.exists(os.path.join(PathSaveFinals, "PlotsTrain")):
        os.mkdir(os.path.join(PathSaveFinals, "PlotsTrain"))
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))
    print("Saved plot loss in: " + os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))




# %%
# if Train:
#     weight_for_0 = (1 / np.sum(ytrainOrig[:,0]))*(ytrainOrig.shape[0])/3.0
#     weight_for_1 = (1 / np.sum(ytrainOrig[:,1]))*(ytrainOrig.shape[0])/3.0
#     weight_for_2 = (1 / np.sum(ytrainOrig[:,2]))*(ytrainOrig.shape[0])/3.0

#     class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
#     # class_weight = {"0": {0: weight_for_0}, "1": {1: weight_for_1}, "2": {2:weight_for_2}}
#     # print(class_weight)

#     # { 'output1': {0: ratio_1 , 1: ratio_2} , 'output2': {0: ratio_3 , 1: ratio_4}}
# %%
# if Train:
# #     history = model.fit([xtrainOrig, xtrainLog, xtrainOrig], ytrainOrig, validation_data=([xval_Orig, xval_Log,xval_Orig], yval), epochs=epochs, shuffle=True, class_weight=class_weight)

#     history = model.fit([xtrainOrig, xtrainLog, xtrainOrig], [ytrainOrig[:,0], ytrainOrig[:,1], ytrainOrig[:,2]], validation_data=([xval_Orig, xval_Log,xval_Orig], [yval[:,0], yval[:,1], yval[:,2]]), epochs=epochs, shuffle=True)#, class_weight=class_weight)
# #     model = model.export_model()
# #     print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>


#     try:
#         model.save(os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size)), save_format="tf")
#     except:
#         model.save(os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size) + ".h5"))
        
#     # save history
#     hist = dict()
#     for i in history.history.keys():
#         hist.update({i: history.history[i]})

#     with open(os.path.join(PathSaveFinals, "history.pkl"), 'wb') as f:
#         pickle.dump(hist, f)
    



# %%
# model.summary()
# for layer in model.layers:
#     print(layer.get_config())


# %%
# if Train:
#     if not os.path.exists(os.path.join(PathSaveFinals, "PlotsTrain")):
#         os.mkdir(os.path.join(PathSaveFinals, "PlotsTrain"))
#     plt.figure()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.legend(['train', 'val'], loc='upper left')
#     plt.savefig(os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))
#     print("Saved plot loss in: " + os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))


# %%
if not Train:
    if not os.path.exists(os.path.join(PathSaveFinals, "PlotsTrain")):
        os.mkdir(os.path.join(PathSaveFinals, "PlotsTrain"))
    #xtrainOrig = xtrain_Orig
    #xtrainLog = xtrain_Log
    if PredFinalModel:
        model = load_model(os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size)))
    else:
        model = load_model(os.path.join(PathSaveFinals,"best_model"))
        
    print(os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size)))
    from sklearn import metrics

    if len(G) > 1:
        predicted_y = model.predict([xtest_Orig, xtest_Log])#, use_multiprocessing=True, batch_size=batch_size)
    else:
        predicted_y = model.predict([xtest_Orig, xtest_Log])

    print(predicted_y)
    print(predicted_y[0].shape)
    print(len(predicted_y))
    print(ytest.shape)

    roc_data = dict()
    auc = dict()
    for i in range(nb_classes):
        roc_data[i] = metrics.roc_curve(ytest[:, i], predicted_y[:,i])
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
        APS_data[i] = metrics.precision_recall_curve(ytest[:, i], predicted_y[:,i])
        APS[i] = metrics.average_precision_score(ytest[:, i], predicted_y[:,i])

    f, ax = plt.subplots(figsize=[9, 6])
    ax.plot(APS_data[0][0], APS_data[0][1], 'k-', label='class 0, AUC = {:4.2f}'.format(APS[0]))
    ax.plot(APS_data[1][0], APS_data[1][1], 'b-', label='class 1, AUC = {:4.2f}'.format(APS[1]))
    ax.plot(APS_data[2][0], APS_data[2][1], 'r-', label='class 2, AUC = {:4.2f}'.format(APS[2]))
    ax.legend(loc='lower right')
    f.savefig(os.path.join(PathSaveFinals, "Precision_recall.png"))

    TRUEY = np.float32(ytest)
    print(TRUEY.shape)
    TRUEY = np.argmax(TRUEY, axis=1)#[np.argmax(p, axis=1) for i,p in enumerate(TRUEY)]
    print(TRUEY.shape)

    PREDY = np.float32(predicted_y)
#     print(PREDY.reshape((PREDY.shape[1], PREDY.shape[0])).shape
#     PREDY = np.argmax(PREDY.reshape((PREDY.shape[1], PREDY.shape[0])), axis=1) #[np.argmax(p, axis=1) for i, p in enumerate(PREDY)]
    PREDY = np.argmax(PREDY, axis=1) #[np.argmax(p, axis=1) for i, p in enumerate(PREDY)]
    print(PREDY.shape)
    
    # print(len(TRUEY)) # list len 605
    # print(len(PREDY)) # list len 605
    # print(type(TRUEY[0])) # <class 'numpy.int64'>
    # print(type(PREDY[0])) # <class 'numpy.int64'>
    # print(TRUEY[0])
    # print(PREDY[0])

    CMatrix = {'Test':dict()}
    CMatrix['Test'] = cm_analysis(y_true=TRUEY,
                                         y_pred=PREDY,
                                         filename=os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_'),
                                         classes=Ylevels,
                                         labels=np.arange(0,3,1),
                                         figsize=(20,20),
                                         cmap='Purples') 
    print("Saved plot CM in: " + os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_.png'))
    print("Saved plot loss in: " + os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))

    # model.summary()
    # for layer in model.layers:
    #     print(layer.get_config())

from contextlib import redirect_stdout
with open(os.path.join(PathSaveFinals, 'modelLOGandOrigCONCATsummary.txt'), 'w') as f:
    with redirect_stdout(f):
        model.summary()
        


# %%
# from sklearn import metrics

# if len(G) > 1:
#     predicted_y = model.predict([xtest_Orig, xtest_Log, xtest_Orig])#, use_multiprocessing=True, batch_size=batch_size)
# else:
#     predicted_y = model.predict([xtest_Orig, xtest_Log, xtest_Orig])

# print(predicted_y)
# print(predicted_y[0].shape)
# print(len(predicted_y))
# print(ytest.shape)

# roc_data = dict()
# auc = dict()
# for i in range(nb_classes):
#     roc_data[i] = metrics.roc_curve(ytest[:, i], predicted_y[i])
#     auc[i] = metrics.auc(roc_data[i][0], roc_data[i][1])

# f, ax = plt.subplots(figsize=[9, 6])
# ax.plot(roc_data[0][0], roc_data[0][1], 'k-', label='class 0, AUC = {:4.2f}'.format(auc[0]))
# ax.plot(roc_data[1][0], roc_data[1][1], 'b-', label='class 1, AUC = {:4.2f}'.format(auc[1]))
# ax.plot(roc_data[2][0], roc_data[2][1], 'r-', label='class 2, AUC = {:4.2f}'.format(auc[2]))
# # ax.plot([0, 1], [0, 1], 'g--')
# ax.legend(loc='lower right')
# f.savefig(os.path.join(PathSaveFinals, "RocClasses.png"))

# APS_data = dict()
# APS = dict()
# for i in range(nb_classes):
#     # APS_data[i] = metrics.roc_curve(y_test, preds)
#     APS_data[i] = metrics.precision_recall_curve(ytest[:, i], predicted_y[i])
#     APS[i] = metrics.average_precision_score(ytest[:, i], predicted_y[i])

# f, ax = plt.subplots(figsize=[9, 6])
# ax.plot(APS_data[0][0], APS_data[0][1], 'k-', label='class 0, AUC = {:4.2f}'.format(APS[0]))
# ax.plot(APS_data[1][0], APS_data[1][1], 'b-', label='class 1, AUC = {:4.2f}'.format(APS[1]))
# ax.plot(APS_data[2][0], APS_data[2][1], 'r-', label='class 2, AUC = {:4.2f}'.format(APS[2]))
# ax.legend(loc='lower right')
# f.savefig(os.path.join(PathSaveFinals, "Precision_recall.png"))
    
# TRUEY = np.float32(ytest)
# print(TRUEY.shape)
# TRUEY = np.argmax(TRUEY, axis=1)#[np.argmax(p, axis=1) for i,p in enumerate(TRUEY)]
# print(TRUEY.shape)

# PREDY = np.float32(predicted_y)
# print(PREDY.reshape((PREDY.shape[1], PREDY.shape[0])).shape)
# PREDY = np.argmax(PREDY.reshape((PREDY.shape[1], PREDY.shape[0])), axis=1) #[np.argmax(p, axis=1) for i, p in enumerate(PREDY)]
# print(PREDY.shape)
# # print(len(TRUEY)) # list len 605
# # print(len(PREDY)) # list len 605
# # print(type(TRUEY[0])) # <class 'numpy.int64'>
# # print(type(PREDY[0])) # <class 'numpy.int64'>
# # print(TRUEY[0])
# # print(PREDY[0])

# CMatrix = {'Test':dict()}
# CMatrix['Test'] = cm_analysis(y_true=TRUEY,
#                                      y_pred=PREDY,
#                                      filename=os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_'),
#                                      classes=Ylevels,
#                                      labels=np.arange(0,3,1),
#                                      figsize=(20,20),
#                                      cmap='Purples') 
# print("Saved plot CM in: " + os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_.png'))
# print("Saved plot loss in: " + os.path.join(PathSaveFinals, "PlotsTrain", "PlotLossHistory.png"))


# %%





# %%





# %%




