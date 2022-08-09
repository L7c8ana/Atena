#!/usr/bin/env python
# coding: utf-8

# Classificou bem classe 2 com 100 %


# !pip install autokeras
# !pip install PyQt5
# !pip install scikit-image
# !apt-get update
# !apt-get install ffmpeg libsm6 libxext6  -y
# !pip install seaborn
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

import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics

# In[2]:


#get_ipython().system('ls -l /tf/DataProjects/NewBoreholeMarcio/')


# In[3]:


# G = ["8", "9"]
G = ["1"]
stringGPUs = []
for i in G:
    GPS = "GPU:" + i
    stringGPUs.append(GPS)


print(stringGPUs)
print(",".join(G))
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(G)

if len(G) > 1:
    strategy = tf.distribute.MirroredStrategy(devices=stringGPUs, cross_device_ops=tf.distribute.NcclAllReduce())
else:
    strategy = tf.distribute.OneDeviceStrategy(device=stringGPUs[0])

print("Number of devices: {}".format(strategy.num_replicas_in_sync))


# In[4]:


# !mv /tf/DataProjects/NewBoreholeMarcio/dataNormW_0.pkl /tf/DataProjects/NewBoreholeMarcio/dataNormW1_W3_W9.pkl


# In[5]:


data_dirSave = "/tf/DataProjects/NewBoreholeMarcio"
#import sys
#sys.path.insert(0, os.path.join(data_dirSave,"EntregaPetroDissolucao"))
from data_loader import load_dataClassNewData #load_dataClassNewMarcioTrain#, load_dataClass2Borehole
# data_dirSave = "/tf/DataProjects/NewBoreholeMarcio"
DataIMGChannel = [0,1,2]
AKCustom = False
ClssDA = 2
dataPred = False
Train = False
max_trials = 5
epochs = 100
batch_size = 32 * len(G)
window_size = 41
DataModel = None # [os.path.join(data_dirSave, "W1_5315-5738m_AMP.npy")]#, os.path.join(data_dirSave, "W3_5314-5883m_AMP_RAW.npy"), os.path.join(data_dirSave, "W9_AMP.npy")]
# sep = [";", ";", ","]#, ";", ",", ",", ","]#[",", ",", ";", ";", ";", ";", ",", ",", ",", ",", ","]
IDx1Borehole = ["W2"]#, "W2", "W3", "W5", "W6", "W8", "W9", "W10", "W11", "W12", "W13"] #["W1", "W3", "W9"]#, "W6", "W8", "W10", "W11", "W12"]#["W1", "W2", "W3", "W5", "W6", "W8", "W9", "W10", "W11", "W12", "W13"]
IDx2Borehole = ["_5485_5915m_AMP"]#, "_5485_5915m_AMP", "_5314-5883m_AMP_RAW", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP"] #["_5315-5738m_AMP", "_5314-5883m_AMP_RAW", "_AMP"]#, "_AMP", "_AMP", "_AMP", "_AMP", "_AMP"]#["_5315-5738m_AMP", "_5485_5915m_AMP", "_5314-5883m_AMP_RAW", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP", "_AMP"]
# ft = []
ReloadPKL = "None"
nb_classes = 3
data_dirIMG = []
data_dirClass = []
titlefoldsave = "TRAIN3BestModelsCLASSESinALLW"
if Train:
    PathSaveFinals = os.path.join(data_dirSave, "ResultCONCTRAINWS" + str(window_size) + titlefoldsave)
    DataModelTrain = PathSaveFinals
else:
    PathSaveFinals = os.path.join(data_dirSave, "ResultCONCTRAINWS" + str(window_size) + titlefoldsave)
    DataModelTrain = os.path.join(data_dirSave, "ResultCONCTRAINWS" + str(window_size) + "TRAIN3BestModelsCLASSESinALLW")

foldermodel0 = os.path.join(data_dirSave, "ResultTRAINWS41AKDissolutionClipPercentileLog0W1")
foldermodel1 = os.path.join(data_dirSave, "ResultTRAINWS41AKDissolutionClipPercentileOrig1W1")
foldermodel2 = os.path.join(data_dirSave, "ResultTRAINWS41AKDissolutionClipPercentileGauss2W1")    
if not os.path.exists(PathSaveFinals):
    os.mkdir(PathSaveFinals)

for s, n, m in zip(range(len(IDx2Borehole)),IDx1Borehole, IDx2Borehole):
    h = n + m
    data_dirIMG0 = os.path.join(data_dirSave, h + ".csv")
    data_dirClass0 = os.path.join(data_dirSave, n + "_FACIESRODAAN.csv")
    data_dirIMG.append(data_dirIMG0)
    data_dirClass.append(data_dirClass0)
   
data_norm = load_dataClassNewData(ReloadPKL=ReloadPKL, data_dirIMG=data_dirIMG, data_dirClass=data_dirClass, data_dirSave=data_dirSave,
                                           MakeChannels=True, dataPred=dataPred, DataTrain=DataModelTrain, Train=Train, LabelWell=IDx1Borehole)

# Nao por essa celula antes do load
#import sys
#sys.path.insert(0, os.path.abspath(os.path.join(data_dirSave,'EntregaPetroDissolucao','BoreholeTools')))
#print("Folder Actual: " + os.getcwd())
#from SlidingDataGenerator import SlidingDataGenerator
#from BoreholeImporter import Data
# from data_loader import load_dataClass, load_data

from utils import ismember, cm_analysis, divideblocksizeTensor

# In[7]:


def uniqueStr(l):
    u = list(set(l))
    i = []
    for c in l:
        i.append(int(np.float32(c)))
        
    return u,i


# In[8]:


#orig = data_norm['Acoustic'][:,:,0].reshape(data_norm['Acoustic'].shape[0], data_norm['Acoustic'].shape[1], 1) 
data_norm['Acoustic'][:,:,1] = np.nan_to_num(data_norm['Acoustic'][:,:,1], nan=0.0)
#log = data_norm['Acoustic'][:,:,1].reshape(orig.shape[0], orig.shape[1], 1) 
#print(log.max(), log.min())
#print(orig.max(), orig.min())
# plt.figure()
# plt.imshow(orig[0:100,:])
# plt.show()
# plt.figure()
# plt.imshow(log[0:100,:])
# plt.show()
#gauss = data_norm['Acoustic'][:,:,2].reshape(orig.shape[0], orig.shape[1], 1) 
#print(gauss.shape)


# In[9]:

X = data_norm['Acoustic'][:,:,np.array(DataIMGChannel)]#.reshape(orig.shape[0], orig.shape[1], 1)
#X = X.reshape(X.shape[0], X.shape[1], 1)
print(X.shape)

# print(data_norm['FDISSOL'])
# print(type(data_norm['FDISSOL']))
Ylab = data_norm['FDISSOL']


# In[10]:


Ylevels, Ynum = uniqueStr(Ylab)

#nb_classes = len(Ylevels)

__M__ = np.shape(X)[0]



LabelBH = "_".join(IDx1Borehole)
if Train:
    train_split, test_split, validation_split = 0.4, 0.2, 0.4
else:
    train_split, test_split, validation_split = 0.0, 1.0, 0.0

xtrain, Ytrain, xtest, Ytest, xval, Yval, ids = divideblocksizeTensor(X, Ynum, window_size, train_split, test_split, validation_split,LabelBH)
#xtrain_Log, Ytrain, xtest_Log, Ytest, xval_Log, Yval, ids = divideblocksizeTensor(log, Ynum, window_size, train_split, test_split, validation_split)
#xtrain_Gauss, Ytrain, xtest_Gauss, Ytest, xval_Gauss, Yval, ids = divideblocksizeTensor(gauss, Ynum, window_size, train_split, test_split, validation_split)
       
print(xtrain.shape, xtest.shape, xval.shape)
print(Ytrain.shape, Ytest.shape, Yval.shape)


ytrain = to_categorical(Ytrain, num_classes=nb_classes)
ytest = to_categorical(Ytest, num_classes=nb_classes)
yval = to_categorical(Yval, num_classes=nb_classes)
print("#############")
print(ytrain.shape, ytest.shape, yval.shape)
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

# Dicionario para ativacao das funcoes
transformations = {'Horizontal flip': h_flip,
                   'Vertical flip': v_flip,
                  }
#############################################################################################
IDXCasos = [n for n, cl in enumerate(ytrain1label) if cl == 1 or cl == 2]
print(len(IDXCasos))

CLS = []
ccc = 0
DataNormAugOrig = np.zeros((len(IDXCasos)*len(transformations),xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]))
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
        DataNormAugOrig[ccc] = TfI
        ccc += 1
        
        CLS.append(clss)

# CLS = np.array(CLS)
# print(CLS.shape)
ytrainDA = np.array(CLS)#tf.keras.utils.to_categorical(CLS, num_classes=3, dtype='float32')
print(ytrainDA.shape)
if Train:
    xtrain = np.concatenate((xtrain,DataNormAugOrig), axis=0)# classifica bem a classe 2
    ytrain = np.concatenate((ytrain,ytrainDA), axis=0)# classifica bem a classe 2
#########################################################################################################

    print(xtrain.shape)
    print(ytrain.shape)

model0 = load_model(os.path.join(foldermodel0, "model_autokerasWS" + str(window_size)))# gauss
model1 = load_model(os.path.join(foldermodel1, "model_autokerasWS" + str(window_size)))#log 
model2 = load_model(os.path.join(foldermodel2, "model_autokerasWS" + str(window_size)))#orig

if Train:
    with strategy.scope():
        #model0 = load_model(os.path.join(foldermodel0, "model_autokerasWS" + str(window_size)))# gauss
        #model1 = load_model(os.path.join(foldermodel1, "model_autokerasWS" + str(window_size)))#log 
        #model2 = load_model(os.path.join(foldermodel2, "model_autokerasWS" + str(window_size)))#orig
        #model0.summary()

        #model1.summary()
        #model2.summary()

    #initializer0 = tensorflow.keras.initializers.GlorotUniform()

    #def reinitialize_layer(model, initializer, layer_name, trainable=False):
    #    layer = model.get_layer(layer_name)
    #    layer.trainable = trainable
    #    layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
    #    return model


                          
    #for n, i in enumerate(model0.layers):
    #    if 0:#n == 6:
    #        for j in model0.layers[n].layers:
    #             print(j.name)
    #            if j.name[-3:] == "_bn":# or j.name[-12:] == "ormalization":
    #                if "kernel_initializer" in j.get_config():
    #                    initializer = j.kernel_initializer
    #                    reinitialize_layer(model1.layers[n], initializer, j.name, trainable=True)
    #                else:
    #                    reinitialize_layer(model1.layers[n], initializer0, j.name, trainable=True)
        #else:
        #    if "kernel_initializer" in model1.layers[n].get_config():
        #        initializer = model1.layers[n].kernel_initializer
        #        reinitialize_layer(model1, initializer, model1.layers[n].name, trainable=True)
        #    else:
        #        reinitialize_layer(model1, initializer0, model1.layers[n].name, trainable=True)
                
    #for n, i in enumerate(model1.layers):
    #    if 0:#n == 5:
    #        for j in model1.layers[n].layers:
    #             print(j.name)
    #            if j.name[-3:] == "_bn":#or j.name[-12:] == "ormalization":
    #                 print(j.name[-3:])
    #                 print(j.name[-12:])
    #                
    #                if "kernel_initializer" in j.get_config():
    #                    initializer = j.kernel_initializer
    #                    reinitialize_layer(model2.layers[n], initializer, j.name, trainable=True)
    #                else:
    #                    reinitialize_layer(model2.layers[n], initializer0, j.name, trainable=True)


        #else:
        #    if "kernel_initializer" in model2.layers[n].get_config():
        #        initializer = model2.layers[n].kernel_initializer
        #        reinitialize_layer(model2, initializer, model2.layers[n].name, trainable=True)
        #    else:
        #        reinitialize_layer(model2, initializer0, model2.layers[n].name, trainable=True)
                
    #for n, i in enumerate(model2.layers):
    #    if 0:#n == :
    #        for j in model2.layers[n].layers:
    #             print(j.name)
    #            if j.name[-3:] == "_bn":#or j.name[-12:] == "ormalization":
    #                 print(j.name[-3:])
    #                 print(j.name[-12:])
                    
    #                if "kernel_initializer" in j.get_config():
    #                    initializer = j.kernel_initializer
    #                    reinitialize_layer(model3.layers[n], initializer, j.name, trainable=True)
    #                else:
    #                    reinitialize_layer(model3.layers[n], initializer0, j.name, trainable=True)


        #else:
        #    if "kernel_initializer" in model3.layers[n].get_config():
        #        initializer = model3.layers[n].kernel_initializer
        #        reinitialize_layer(model3, initializer, model3.layers[n].name, trainable=True)
    #    else:
    #        reinitialize_layer(model3, initializer0, model3.layers[n].name, trainable=True)
                                

    ######################
    # print(model1.layers[-1].get_config())
    # print(model2.layers[-1].get_config())
    # print(model3.layers[-1].get_config())
        #print(model0.optimizer.get_config())
        #print(model1.optimizer.get_config())
        #print(model2.optimizer.get_config())
    # print(model1.loss())
    # print(model2.loss.get_config())
    # print(model3.loss.get_config())
    #########
        input_shape = (window_size,xtrain.shape[2],1)
    # print(input_shape)
        input_ = Input(shape=input_shape, name="img")
        model0._name = model0.name + "_0"
        model1._name = model1.name + "_1"
        model2._name = model2.name + "_2"
    # 
    # inpSeparate1 = Lambda(lambda x: x[0])(input_)
        model0_ = model0#(input_)
    # inpSeparate2 = Lambda(lambda x: x[1])(input_)
        model1_ = model1#(input_)
    # inpSeparate3 = Lambda(lambda x: x[2])(input_)
        model2_ = model2#(input_)
        for n, layer in enumerate(model0.layers):
            layer._name = layer.name + str("_0")
        
        for n, layer in enumerate(model1.layers):
            layer._name = layer.name + str("_1")
        
    for n, layer in enumerate(model2.layers):
        layer._name = layer.name + str("_2")
    conc = Concatenate()([model0.layers[-1].output, model1.layers[-1].output, model2.layers[-1].output])
    dense = Dense(3, name="dense012")(conc)
    output = Dense(3, activation='softmax', name="output012")(dense)
    #output2 = Dense(1, activation='sigmoid', name="1")(model2.layers[-1].output)
    #output3 = Dense(1, activation='sigmoid', name="2")(model3.layers[-1].output)
    model = Model([model0.input, model1.input, model2.input], output)
    #model = Model([model1.input, model2.input, model3.input], [output1, output2, output3])

    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_crossentropy"])
    #config = model.to_json()
    #loaded_model = tensorflow.keras.models.model_from_json(config)

    #losses = {xn: "categorical_crossentropy" for n, xn in enumerate(self.outputs)}

    #metrics = {xn: "categorical_crossentropy" for n, xn in enumerate(self.outputs)}

    #lossWeights = {xn: 1.0 for n, xn in enumerate(self.outputs)}
    INIT_LR =  1e-05 #/ 10
    optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR)#, decay=INIT_LR / self.epochs)#, epsilon=1e-05)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]) #loss=losses, loss_weights=lossWeights, optimizer=tf.keras.optimizers.Adam(),#learning_rate=INIT_LR),#, decay=INIT_LR / self.epochs, epsilon=1e-05),
              # metrics=metrics) #epsilon=1e-07
    print(model.summary())


   

# In[20]:


    weight_for_0 = (1 / np.sum(ytrain[:,0]))*(ytrain.shape[0])/3.0
    weight_for_1 = (1 / np.sum(ytrain[:,1]))*(ytrain.shape[0])/3.0
    weight_for_2 = (1 / np.sum(ytrain[:,2]))*(ytrain.shape[0])/3.0

    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
    print(class_weight)


# In[ ]:

    ytrainN = np.zeros(ytrain.shape)
    ytrainN[:,0] = ytrain[:,1]
    ytrainN[:,1] = ytrain[:,0]
    ytrainN[:,2] = ytrain[:,2]

    yvalN = np.zeros(yval.shape)
    yvalN[:,0] = yval[:,1]
    yvalN[:,1] = yval[:,0]
    yvalN[:,2] = yval[:,2]

    ytestN = np.zeros(ytest.shape)
    ytestN[:,0] = ytest[:,1]
    ytestN[:,1] = ytest[:,0]
    ytestN[:,2] = ytest[:,2]

    filepathBestModel = os.path.join(PathSaveFinals, "model_autokerasBestModel")
    filenamecsv1 = os.path.join(PathSaveFinals, "CSVLOGhistoryTrain.csv")
    callback = [tf.keras.callbacks.CSVLogger(filenamecsv1, separator=',', append=False),
            tf.keras.callbacks.ModelCheckpoint(
                                    filepathBestModel,
                                    monitor='val_accuracy',
                                    mode='max')
            ]
    sh0T,sh1T,sh2T,sh3T = xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]
    sh0V,sh1V,sh2V,sh3V = xval.shape[0], xval.shape[1], xval.shape[2], xval.shape[3]
    history = model.fit([xtrain[:,:,:,1].reshape((sh0T,sh1T,sh2T,1)),xtrain[:,:,:,0].reshape((sh0T,sh1T,sh2T,1)),xtrain[:,:,:,2].reshape((sh0T,sh1T,sh2T,1))], ytrain, validation_data=([xval[:,:,:,1].reshape((sh0V,sh1V,sh2V,1)), xval[:,:,:,0].reshape((sh0V,sh1V,sh2V,1)),xval[:,:,:,2].reshape((sh0V,sh1V,sh2V,1))], yval), epochs=epochs, class_weight=class_weight, callbacks=callback)
    #model = model.export_model()
    print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

    
    try:
        model.save(os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size)), save_format="tf")
    except:
        model.save(os.path.join(PathSaveFinals,"model_autokerasWS" + str(window_size) + ".h5"))


# In[ ]:


# model = auto_model.export_model()


# In[ ]:


    model.summary()


# In[ ]:


    for layer in model.layers:
        print(layer.get_config())





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


# In[ ]:

else:
    sh0tt,sh1tt,sh2tt,sh3tt = xtest.shape[0], xtest.shape[1], xtest.shape[2], xtest.shape[3]
    model = load_model(os.path.join(DataModelTrain,"model_autokerasBestModel"))
    if len(G) > 1:
        predicted_y = model.predict([xtest[:,:,:,1].reshape((sh0tt,sh1tt,sh2tt,1)),xtest[:,:,:,0].reshape((sh0tt,sh1tt,sh2tt,1)),xtest[:,:,:,2].reshape((sh0tt,sh1tt,sh2tt,1))], use_multiprocessing=True, batch_size=batch_size)
    else:
        predicted_y = model.predict([xtest[:,:,:,1].reshape((sh0tt,sh1tt,sh2tt,1)),xtest[:,:,:,0].reshape((sh0tt,sh1tt,sh2tt,1)),xtest[:,:,:,2].reshape((sh0tt,sh1tt,sh2tt,1))])

    roc_data = dict()
    auc = dict()
    optth = dict()
    for i in range(nb_classes):
        roc_data[i] = metrics.roc_curve(ytest[:, i], predicted_y[:, i])
        auc[i] = metrics.auc(roc_data[i][0], roc_data[i][1])
        fpr, tpr, thr = metrics.roc_curve(ytest[:, i], predicted_y[:, i])
        i_opt = np.argmin([np.linalg.norm((1 - ti, fi)) for (ti, fi) in zip(tpr, fpr)])
        optth[i] = thr[i_opt]
        print(auc[i])

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
        APS_data[i] = metrics.precision_recall_curve(ytest[:, i], predicted_y[:, i])
        APS[i] = metrics.average_precision_score(ytest[:, i], predicted_y[:, i])

    f, ax = plt.subplots(figsize=[9, 6])
    ax.plot(APS_data[0][0], APS_data[0][1], 'k-', label='class 0, AUC = {:4.2f}'.format(APS[0]))
    ax.plot(APS_data[1][0], APS_data[1][1], 'b-', label='class 1, AUC = {:4.2f}'.format(APS[1]))
    ax.plot(APS_data[2][0], APS_data[2][1], 'r-', label='class 2, AUC = {:4.2f}'.format(APS[2]))
    ax.legend(loc='lower right')
    f.savefig(os.path.join(PathSaveFinals, "Precision_recall.png"))
    
    TRUEY = np.float32(ytest)
    TRUEY = np.argmax(TRUEY, axis=1)#[np.argmax(p, axis=1) for i,p in enumerate(TRUEY)]
    #PREDY = np.float32(predicted_y)
    #PREDY = np.argmax(PREDY, axis=1) #[np.argmax(p, axis=1) for i, p in enumerate(PREDY)]
    #PREDY = np.zeros((predicted_y.shape[0],3))
    #for i in range(3):
    #    for n,cl in enumerate(predicted_y[:,i]):
    #        if cl < optth[i]:
    #            PREDY[n,i] = 1.0
    PREDY = np.float32(predicted_y)
    PREDY = np.argmax(PREDY, axis=1)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(TRUEY,PREDY, normalize="true"))#, labels=labels)
    np.savetxt(os.path.join(PathSaveFinals, "CM.npy"), confusion_matrix(TRUEY,PREDY,normalize="true"), delimiter=",")
    kkkkk
    CMatrix = {'Test':dict()}
    CMatrix['Test'] = cm_analysis(y_true=TRUEY,
                                         y_pred=PREDY,
                                         filename=os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_'),
                                         classes=Ylevels,
                                         labels=np.arange(0,3,1),
                                         figsize=(20,20),
                                         cmap='Purples') 
    print("Saved plot CM in: " + os.path.join(PathSaveFinals, "PlotsTrain",'ConfMatrix_Test_.png'))


# In[ ]:





# In[ ]:





# In[ ]:




