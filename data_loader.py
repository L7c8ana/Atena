import os
import scipy.io
import numpy as np
import pandas as pd
import pickle
from scipy import interpolate
# from _io import save_results2, load_results2
# from SlidingGenerator import SlidingGenerator
#from intervals import divide_blocks
from scipy import ndimage
import matplotlib.pyplot as plt
import tensorflow as tf
import math


def save_resultsPKL(results, path = None):
    with open(path,'wb') as f:
        pickle.dump(results,f)

""" function to load results """
def load_resultsPKL(path=None):
    with open(path,'rb') as f:
        results = pickle.load(f)
    return results

def MinMaxScaler(array, scale=[0, 1]):
    array_max = array.max()
    array_min = array.min()
    return scale[0] + (((array - array_min) * (scale[1] - scale[0])) / (array_max - array_min))


def load_results2(path):
    with open(path, 'rb') as f:
        results = pickle.load(f)

    return results


# +
def load_dataClassNewData(ReloadPKL=True, data_dirIMG=None, data_dirClass=None, data_dirSave=None, MakeChannels=None, dataPred=True, DataTrain=None, Train=None, LabelWell=None):
    # inputs : CSVs 
    # outputs : Dict with : Array Data acoustic normalize 0-1 float32 (U) and array labels categorical FDISSOL - int32
    #     U_ALL = np.array([])
    U_ALL = []
    U_ALLNoNorm = []
    DepthAMP_ALL = []
    FDISSOL_ALL = []
    # filePKLMinMaxNoNorm
    # filePKLMinMax
    LabelWell = "_".join(LabelWell)
    print("Folder save: " + data_dirSave)

    #     print("############################################################################################")
    #     print("#### LOAD BOREHOLE: " + fileImg)
    #     print("############################################################################################")
    #     filePKL = os.path.join(data_dirSave, "data" + LabelWell + ".pkl")
    if Train:
        #filePKLNorm = os.path.join(DataTrain, "dataNorm_Train.pkl")
        filePKLMinMax = os.path.join(DataTrain, "dataNormMinMax_Train.pkl")
        #filePKLNoNorm = os.path.join(DataTrain, "dataNoNorm_Train.pkl")
        filePKLMinMaxNoNorm = os.path.join(DataTrain, "dataNoNormMinMax_Train.pkl")
    else:
        #filePKLNorm = os.path.join(DataTrain, "dataNorm_Train.pkl")
        filePKLMinMax = os.path.join(DataTrain, "dataNormMinMax_Train.pkl")
        #filePKLNoNorm = os.path.join(DataTrain, "dataNoNorm_Train.pkl")
        filePKLMinMaxNoNorm = os.path.join(DataTrain, "dataNoNormMinMax_Train.pkl")

    if 0:#not ReloadPKL:
        if os.path.isfile(filePKLNorm): # and os.path.isfile(filePKLMinMax):#
            print("Load files Dataframe in " + filePKLNorm)
            data_norm = load_resultsPKL(path = filePKLNorm)
            print("Data Shape Load in PKL:")
            print("U_ALL shape", data_norm["Acoustic"].shape)
            print("U max norm", data_norm["Acoustic"].max())
            print("U max norm", data_norm["Acoustic"].min())
            print("U max norm channel 0", data_norm["Acoustic"][:,:,0].max())
            print("U max norm channel 0", data_norm["Acoustic"][:,:,0].min())
            print("U max norm channel 1", data_norm["Acoustic"][:,:,1].max())
            print("U max norm channel 1", data_norm["Acoustic"][:,:,1].min())
            print("U max norm channel 2", data_norm["Acoustic"][:,:,2].max())
            print("U max norm channel 2", data_norm["Acoustic"][:,:,2].min())
            print("Depth AMP shape", np.array(data_norm["Depth"]).shape)
            print("Class shape", np.array(data_norm["FDISSOL"]).shape)
        
    else:

        for i, j in enumerate(data_dirIMG):
            fileImg = data_dirIMG[i][:-4]
            fileClass = data_dirClass[i][:-4]
            ##############
            ##### condição pra verificar se precisa interpo;=lacao
            ##############
            try:
                CatIMG = pd.read_csv(data_dirIMG[i], delimiter=";")
            except:
                CatIMG = pd.read_csv(data_dirIMG[i], delimiter=",")
            print("Loading data Images in: " + data_dirIMG[i])
            #                 print(data_dirIMG[i])
            #                 print(CatIMG)
            try:
                CatClass = pd.read_csv(data_dirClass[i], delimiter=";")
            except:
                CatClass = pd.read_csv(data_dirClass[i], delimiter=",")
            #                 print(CatClassv.shape)

            print("Loading data Classes in: " + data_dirClass[i])
            #                 print(data_dirClass[i])
            #                 print(CatClass)


            CatIMGA = np.array(CatIMG.values)
            #                 print(CatIMGA.shape)
            
            

            U = CatIMGA[1:, 1:-1]
            #                 print(U.shape)

            if U.shape[1] > 180:
                U = CatIMGA[1:, 1:1+180]

            if U.shape[1] < 180:
                U = CatIMGA[1:, 1:]

            U = np.float64(U)
            
                
            #                 print(U.max(), U.min())

            ic = np.float64(-100.00)
            #                 U = np.where(U <= ic, np.min(U), U)
            #                 print(U)

            CatClassA = np.array(CatClass.values)
            if CatClassA.shape[1] < 2:
                try:
                    CatClass = pd.read_csv(data_dirClass[i], delimiter=",")
                except:
                    CatClass = pd.read_csv(data_dirClass[i], delimiter=";")

            CatClassA = np.array(CatClass.values)
            DepthAMP = CatIMGA[1:, 0]

            print(CatClassA.shape)
            print(CatClassA)

            FDISSOL = CatClassA[1:, 1]

            Depth = CatClassA[1:, 0]

            del CatClassA

            print("Data Shape BEFORE interpolation:")
            print("U", U.shape)
            print("Depth AMP", DepthAMP.shape)
            print("Class", FDISSOL.shape)
            print("Depth Dissol", Depth.shape)

            ####### INTERPOLATION
            print("Making Interpolation ...")
            PETRO = {"Depth": Depth, "FDISSOL": FDISSOL}
            """ Now crop depth between in common interval """
            d0 = CatIMGA[1:, 0] # depth data big
            d1 = PETRO["Depth"]
            for v in d0:
                if isinstance(v, str):
                    #                     print(type(v))
                    #                     print(v)
                    d0 = np.float64(CatIMGA[1:, 0]) # depth data big
                    d1 = np.float64(PETRO["Depth"])
                    break

            # d2 = FDISSOL
            dres0 = np.min(np.diff(np.sort(d0)))
            dres1 = np.min(np.diff(np.sort(d1)))

            dres = np.minimum(dres0, dres1)
            dmin = np.maximum(d0.min(), d1.min())
            dmax = np.minimum(d0.max(), d1.max())

            """ Create new depth """
            dref = np.arange(dmin, dmax, dres)

            """ Now crop new data """
            hh = []
            for j in d1:
                if j >= dmin or j <= dmax:
                    hh.append(j)

            PETRO["Depth"] = hh


            """ Crop old data too """
            idxs2keep = (d0 >= dmin) & (d0 <= dmax)
            d0 = d0[idxs2keep]
            U = U[idxs2keep, :]
            DepthAMP = d0

            """ Interpolate new data """
            fP = interpolate.interp1d(PETRO['Depth'], PETRO["FDISSOL"])
            FD = fP(dref)
            FDISSOL = FD

            if U.shape[0] != FDISSOL.shape[0]:
                print("Error")
                assert U.shape[0] != FDISSOL.shape[0], "Error Invalid Operation shape Image no match shape classifications"

            if U.shape[1] != 180:
                assert U.shape[1] != 180, "Error Invalid Operation shape Image no match shape 180 columns"


            """ Define float type """

            U = np.float32(U)
            print("********************")
            print(U.min(), U.max())
            if not Train:
                # filePKLMinMaxNoNorm
                # filePKLMinMax
                #Info_dataTrain = {"DataTrainWell" : ["W1", "W3", "W9"], "channels": ["Acoustic", "Log", "Gauss"] }
                #filePKLMinMaxTrain = os.path.join(DataTrain,"dataNormMinMax_Train.pkl")
                #filePKLMinMaxTrainNoNorm = os.path.join(DataTrain,
                #                                  "dataNoNormMinMax_Train.pkl")
                if os.path.exists(filePKLMinMaxNoNorm) and os.path.exists(filePKLMinMax):
                    minmaxtrain = load_resultsPKL(path=filePKLMinMax)
                    minmaxtrainNoNorm = load_resultsPKL(path=filePKLMinMaxNoNorm)
                    #print(minmaxtrain)
                    #print(minmaxtrainNoNorm)
                    
                else:
                    print("No exists: " + filePKLMinMaxTrainNoNorm)
                    print("No exists: " + filePKLMinMaxTrain)
            
            if Train:
                np.save(os.path.join(DataTrain,"dataBeforePercMinMax_Train.npy"), np.array([U.min(),U.max()]))
                print(U.min(),U.max())
                
            #FDISSOL = np.uint8(FDISSOL)
            if MakeChannels:
                print("Making 3 channels ....")
                if Train:
                    #np.save(os.path.join(DataTrain,"dataBeforePercMinMax_Train.npy"), np.array([U.min(),U.max()]))
                    U = np.nan_to_num(U, nan=0.0)
                    ch_max = np.percentile(U, 99.9)
                    ch_min = -(np.percentile(-U, 99.9))
                    Up = np.clip(U, ch_min, ch_max)
                    print(ch_min, ch_max)
                
                    U = Up
                    
                else:
                    um = np.load(os.path.join(DataTrain,"dataBeforePercMinMax_Train.npy"))
                    #np.save(os.path.join(DataTrain,"dataBeforePercMinMax_Train.npy"), np.array([U.min(),U.max()]))
                    umn = um[0]
                    umx = um[1]
                    U = np.nan_to_num(U, nan=0.0)
                    #U = MinMaxScaler(U, scale=[umn,umx])
                    #ch_max = np.percentile(U, 99.9)
                    #ch_min = -(np.percentile(-U, 99.9))
                    #Up = np.clip(U, ch_min, ch_max)
                    #if dataPred:
                    #U = Up
                
                if Train:
                    bins = np.arange(np.min(U), np.max(U),1)
                    binsp = np.arange(np.min(Up), np.max(Up),1)
                    plt.figure()
                    plt.hist(np.ravel(Up),binsp, color="r", alpha=0.2,label="Up")
                    plt.hist(np.ravel(U), bins, color="b", alpha=0.2, label="U")
                    plt.legend()
                    plt.savefig(os.path.join(DataTrain,"HistCompUandUp_NodataPred.png"))
                    print(os.path.join(DataTrain,"HistCompUandUp_NodataPred.png"))
                    plt.figure()
                    plt.imshow(Up[0:1000], cmap="gray")
                    plt.savefig(os.path.join(DataTrain,"ImgTestNewHist.png"))
                    plt.figure()
                    plt.imshow(U[0:1000], cmap="gray")
                    plt.savefig(os.path.join(DataTrain,"ImgTestOrigHist.png"))
                #U = Up
                
                UO = U.copy()
                UO = np.nan_to_num(UO, nan=0.0)
                #UOV = np.ravel(UO)
                #MIN = np.min([k for k in UOV if (k > ic and not math.isnan(k))])
                #MAX = np.max([k for k in UOV if (k > ic and not math.isnan(k))])
#                     UON = (UO - MIN) / (MIN + MAX)
                if Train:
                    #UO = Up
                    UON = MinMaxScaler(UO, scale=[0, 1])
                else:
                    #UO = Up
                    #UO = np.nan_to_num(UO, nan=0.0)
                    um = np.load(os.path.join(DataTrain,"dataBeforePercMinMax_Train.npy"))
                    umn = um[0]
                    umx = um[1]
                    print(um)
                    
                    UON = U#MinMaxScaler(U, scale=[umn,umx])
                    ch_max = np.percentile(UON, 99.9)
                    ch_min = -(np.percentile(-UON, 99.9))
                    UON = np.clip(UON, ch_min, ch_max)

                    #UON = np.clip(U, minmaxtrainNoNorm["MinMaxAcoustic"][0], minmaxtrainNoNorm["MinMaxAcoustic"][1])
                    UON = MinMaxScaler(UO, scale=[0, 1])
                    #UON = MinMaxScaler(UON, scale=[minmaxtrain["MinMaxAcoustic"][0], minmaxtrain["MinMaxAcoustic"][1]])
                    #UON = np.clip(UO, minmaxtrain["MinMaxAcoustic"][0], minmaxtrain["MinMaxAcoustic"][1])
                    

                print("Max and Min original images normalize:")
                print(UON.max(), UON.min())
                
                if Train:
                    Ulogp = np.log10(Up) 
                    Ulogp = np.nan_to_num(Ulogp, nan=0.0)
                    Ulog = np.log10(U)
                    Ulog = np.nan_to_num(Ulog, nan=0.0)
                    #UlogV = np.ravel(Ulog)
                    UlogN = MinMaxScaler(Ulogp, scale=[0, 1])

                else:
                    UON = U#MinMaxScaler(U, scale=[umn,umx])
                    ch_max = np.percentile(UON, 99.9)
                    ch_min = -(np.percentile(-UON, 99.9))
                    UON = np.clip(UON, ch_min, ch_max)
                    Ulog = np.log10(UON)
                    Ulog = np.nan_to_num(Ulog, nan=0.0)
                    #UlogN = np.clip(Ulog, minmaxtrain["MinMaxLog"][0], minmaxtrain["MinMaxLog"][1])
                    #UlogN = MinMaxScaler(UlogN, scale=[minmaxtrain["MinMaxLog"][0], minmaxtrain["MinMaxLog"][1]])
                    UlogN = MinMaxScaler(Ulog, scale=[0, 1])
                    #UlogN = np.clip(UlogN, minmaxtrain["MinMaxAcoustic"][0], minmaxtrain["MinMaxAcoustic"][1])
                
                if Train:
                    bins = np.arange(np.min(Ulog), np.max(U),1)
                    binsp = np.arange(np.min(Ulogp), np.max(Up),1)
                    plt.figure()
                    plt.hist(np.ravel(Ulogp),binsp, color="r", alpha=0.2,label="Up")
                    plt.hist(np.ravel(Ulog),bins, color="b", alpha=0.2,label="U")
                    plt.legend()
                    print(np.min(Ulogp), np.max(Ulogp))
                    plt.savefig(os.path.join(DataTrain,"HistCompUandUlog.png"))
                    plt.figure()
                    plt.imshow(Up[0:1000], cmap="gray")
                    plt.savefig(os.path.join(DataTrain,"ImgTestNewHistLog.png"))
                    plt.figure()
                    plt.imshow(U[0:1000], cmap="gray")
                    plt.savefig(os.path.join(DataTrain,"ImgTestLogHist.png"))

                print("Max and Min log images normalize:")
                print(UlogN.max(), UlogN.min())
                #UG = ndimage.gaussian_gradient_magnitude(U, sigma=1)
                #UG = np.nan_to_num(UG, nan=0.0)
                #UGV = np.ravel(UG)
#                     MIN = np.min([k for k in UGV if (k > ic and not math.isnan(k))])
#                     MAX = np.max([k for k in UGV if (k > ic and not math.isnan(k))])
#                     UGN = (UG - MIN) / (MIN + MAX)
#                     UGN = MinMaxScaler(UG, scale=[0, 1])
                if Train:
                    UG = ndimage.gaussian_gradient_magnitude(Up, sigma=1)
                    UG = np.nan_to_num(UG, nan=0.0)
                    
                    UGN = MinMaxScaler(UG, scale=[0, 1])
                else:
                    UON = U#MinMaxScaler(UO, scale=[umn,umx])
                    ch_max = np.percentile(UON, 99.9)
                    ch_min = -(np.percentile(-UON, 99.9))
                    UON = np.clip(UON, ch_min, ch_max)
                    #UON = np.clip(UON, minmaxtrainNoNorm["MinMaxGauss"][0], minmaxtrainNoNorm["MinMaxGauss"][1])
                    UG = ndimage.gaussian_gradient_magnitude(UON, sigma=1)
                    UG = np.nan_to_num(UG, nan=0.0)
                    #UGN = np.clip(UG, minmaxtrainNoNorm["MinMaxGauss"][0], minmaxtrainNoNorm["MinMaxGauss"][1])
                    UGN = MinMaxScaler(UG, scale=[0, 1])
                    #UGN = MinMaxScaler(UGN, scale=[minmaxtrain["MinMaxGauss"][0], minmaxtrain["MinMaxGauss"][1]])
                    #UGN = np.clip(UGN, minmaxtrain["MinMaxAcoustic"][0], minmaxtrain["MinMaxAcoustic"][1])

                print("Max and Min gauss images normalize:")
                print(UGN.max(), UGN.min())
                UT = np.zeros((UO.shape[0], UO.shape[1], 3))
                UT[:, :, 0] = np.nan_to_num(UO)
                UT[:, :, 1] = np.nan_to_num(Ulog)
                UT[:, :, 2] = np.nan_to_num(UG)
                # UT = np.where(UT<=-900, 1, UT)
                print("Max and Min original images 3 channels after verify nan values:")
                print(UT[:, :, 0].max(), UT[:, :, 0].min())
                print(UT[:, :, 1].max(), UT[:, :, 1].min())
                print(UT[:, :, 2].max(), UT[:, :, 2].min())
                UTN = np.zeros((UO.shape[0], UO.shape[1], 3))
                UTN[:, :, 0] = UON
                UTN[:, :, 1] = UlogN
                UTN[:, :, 2] = UGN
                #                     print(UTN.max(), UTN.min())
                U = UTN.copy()
            else:
                """ Normalize images """
                UT = U
                UT = np.nan_to_num(UT, nan=0.0)
#                     MIN = np.min([k for k in np.ravel(U) if (k > ic and not math.isnan(k))])
#                     MAX = np.max([k for k in np.ravel(U) if (k > ic and not math.isnan(k))])
                #                     U = (U - MIN) / (MIN + MAX)
                if not dataPred:
                    U = MinMaxScaler(U, scale=[0, 1])
                else:
                    U = np.clip(U, minmaxtrainNoNorm[0], minmaxtrainNoNorm[1])
                    U = MinMaxScaler(U, scale=[minmaxtrain["MinMaxAcoustic"][0], minmaxtrain["MinMaxAcoustic"][1]])


                U = U[:, :, None]
            F = []
            for k in FDISSOL:
                F.append(str(k))

            FDISSOL = F
            import random
           
            print(U.shape[0])
            incc = random.randint(1, U.shape[0]-16000)
            
            FDISSOL = FDISSOL[incc:incc+15000]
            U = U[incc:incc+15000,:,:]
            UT = UT[incc:incc+15000,:,:]
            DepthAMP = DepthAMP[incc:incc+15000]
            #             print(U.shape)
            #             print(len(FDISSOL))

            print("Data Shape AFTER interpolation:")
            print("U", U.shape)
            #                 print("U max no norm", UT.max())
            #                 print("U max no norm", UT.min())
            print("U max", U.max())
            print("U max", U.min())
            print("Depth AMP", DepthAMP.shape)
            print("Class", len(FDISSOL))
            print("Depth Dissol", Depth.shape)
            print(i)

            if len(data_dirIMG) > 1:
                if i == 0:
                    U_ALL = U.copy()
                    U_ALLNoNorm = UT.copy()
                    FDISSOL_ALL = FDISSOL.copy()
                    DepthAMP_ALL = DepthAMP.copy()
                else:
                    print("CONCATENA: " + str(i))
                    print(U_ALL.shape) 
                    U_ALL = np.concatenate((np.array(U), np.array(U_ALL)), axis=0)
                    U_ALLNoNorm = np.concatenate((np.array(UT), np.array(U_ALLNoNorm)), axis=0)
                    print(U_ALL.shape)
                    FDISSOL_ALL = np.concatenate((np.array(FDISSOL), np.array(FDISSOL_ALL)))
                    DepthAMP_ALL = np.concatenate((np.array(DepthAMP), np.array(DepthAMP_ALL)))
            else:
                U_ALL = U
                print(U_ALL.shape)
                U_ALLNoNorm = UT
                FDISSOL_ALL = FDISSOL
                DepthAMP_ALL = DepthAMP





        print(U_ALL.shape)
        print("Data Info before Normalize:")
        print("U_ALL shape", U_ALL.shape)
        print("U max norm channel 0", U_ALL[:,:,0].max())
        print("U min norm channel 0", U_ALL[:,:,0].min())
        print("U max norm channel 1", U_ALL[:,:,1].max())
        print("U min norm channel 1", U_ALL[:,:,1].min())
        print("U max norm channel 2", U_ALL[:,:,2].max())
        print("U min norm channel 2", U_ALL[:,:,2].min())
        print("Depth AMP shape", np.array(DepthAMP_ALL).shape)
        print("Class shape", np.array(FDISSOL_ALL).shape)


        print("Data Norm Shape Saved in PKL:")
        print("U_ALL shape", U_ALL.shape)
        print("U max norm", U_ALL.max())
        print("U max norm", U_ALL.min())
        print("Depth AMP shape", np.array(DepthAMP_ALL).shape)
        print("Class shape", np.array(FDISSOL_ALL).shape)

        print("Data No Norm Shape Saved in PKL:")
        print("U_ALL shape", U_ALLNoNorm.shape)
        print("U max norm", U_ALL.max())
        print("U max norm", U_ALL.min())
        print("Depth AMP shape", np.array(DepthAMP_ALL).shape)
        print("Class shape", np.array(FDISSOL_ALL).shape)



    data_norm = {'Acoustic': U_ALL,
                'FDISSOL': FDISSOL_ALL,
                'Depth': np.array(DepthAMP_ALL)
                }
    print(U_ALL.shape)
    data_minmax = {"MinMaxAcoustic": [U_ALL[:, :, 0].min(),U_ALL[:, :, 0].max()],
                "MinMaxLog": [U_ALL[:, :, 1].min(),U_ALL[:, :, 1].max()],
                "MinMaxGauss": [U_ALL[:, :, 2].min(), U_ALL[:, :, 2].max()]
                }

    print(data_minmax)
    """ Save data """
    #             U = np.save(fileImg + '.npy', U)
    #             FDISSOL = np.save(fileClass + '.npy', FDISSOL)
    #             Depth = np.save(fileClass +  '_Depth' + '.npy', DepthAMP)

    #save_resultsPKL(data_norm, path = filePKLNorm)
    save_resultsPKL(data_minmax, path = filePKLMinMax)
    dataNoNorm = {'Acoustic': U_ALLNoNorm,
                 'FDISSOL': FDISSOL_ALL,
                 'Depth': np.array(DepthAMP_ALL)
                 }
    print(U_ALL.shape)
    data_minmaxNoNorm = {"MinMaxAcoustic": [U_ALLNoNorm[:, :, 0].min(), U_ALLNoNorm[:, :, 0].max()],
                   "MinMaxLog": [U_ALLNoNorm[:, :, 1].min(), U_ALLNoNorm[:, :, 1].max()],
                   "MinMaxGauss": [U_ALLNoNorm[:, :, 2].min(), U_ALLNoNorm[:, :, 2].max()]
                   }

    print(data_minmaxNoNorm)
    """ Save data """
    #             U = np.save(fileImg + '.npy', U)
    #             FDISSOL = np.save(fileClass + '.npy', FDISSOL)
    #             Depth = np.save(fileClass +  '_Depth' + '.npy', DepthAMP)

    #save_resultsPKL(dataNoNorm, path=filePKLNoNorm)
    save_resultsPKL(data_minmaxNoNorm, path=filePKLMinMaxNoNorm)
    return data_norm


# -
# --

""" Prepare AE data """
def prepare_AE_data(data, batch_size = 32, sliding_window = 21):

    """ Define inputs (also outputs) """
    inputs = ['Acoustic','Resistivity']

    """ Get number of samples """
    nsamples = data[inputs[0]].__len__()

    sliding_function = {yn: 'none' for yn in inputs}

    """ Define indexes (whole thing) """
    delta = (sliding_window - 1) // 2
    indexes = {yn: np.arange(delta, nsamples - delta) for yn in data}

    """ Create Training/Validation Generator """
    generator = SlidingGenerator(x = {xn: data[xn] for xn in inputs},
                                 sliding_window = sliding_window,
                                 sliding_function = sliding_function,
                                 indexes = indexes,
                                 batch_size = batch_size,
                                 shuffle = True)

    val_gen = SlidingGenerator(x = {xn: data[xn] for xn in inputs},
                               sliding_window = sliding_window,
                               sliding_function = sliding_function,
                               indexes = indexes,
                               batch_size = batch_size,
                               shuffle = False)

    """ Calculate data_shape """
    data_shape = {xn: data[xn].shape[1:] for xn in inputs}

    return generator, val_gen, data_shape


""" Prepare AE data """
def prepare_Class_data(data, batch_size = 32, sliding_window = 21):

    """ Define inputs (also outputs) """
    inputs = ['Acoustic']
    outputs = ['FDISSOL']

    """ Get number of samples """
    nsamples = data[inputs[0]].__len__()

    sliding_function = {yn: 'none' for yn in inputs}

    """ Define indexes (whole thing) """
    delta = (sliding_window - 1) // 2
    indexes = {yn: np.arange(delta, nsamples - delta) for yn in data}

    """ Create Training/Validation Generator """
    generator = SlidingGenerator(x = {xn: data[xn] for xn in inputs},
                                 y = {xn: data[xn] for xn in outputs},
                                 sliding_window = sliding_window,
                                 sliding_function = sliding_function,
                                 indexes = indexes,
                                 batch_size = batch_size,
                                 shuffle = True)

    val_gen = SlidingGenerator(x = {xn: data[xn] for xn in inputs},
                               y = {xn: data[xn] for xn in outputs},
                               sliding_window = sliding_window,
                               sliding_function = sliding_function,
                               indexes = indexes,
                               batch_size = batch_size,
                               shuffle = False)

    """ Calculate data_shape """
    data_shape = {xn: data[xn].shape[1:] for xn in inputs}

    return generator, val_gen, data_shape

""" Prepare Regression data """
def prepare_reg_data2(data_norm, codes, smooth_window = 121, sliding_window = None, batch_size = None,
                     validation_split = None, blocksize = None):

    """ Define inputs (also outputs) """
    outputs = ['PERM']#,'PHIE']
    inputs = ["Acoustic"]

    """ Get number of samples """
    delta = (sliding_window - 1) // 2


    """ Crop data """
    X = {'codes': data_norm[yn][delta:-delta] for yn in inputs} #{'codes': codes_cropped}
    Y = {yn: data_norm[yn][delta:-delta] for yn in outputs}

    """ Calculate data_shape """
    input_shape = {xn: X[xn].shape[1:] for xn in X}
    output_shape = {yn: Y[yn].shape[1:] for yn in Y}

    """ Define indexes """
    train_idxs, _, val_idxs = divide_blocks(Z = X['codes'],
                                            blocksize = blocksize,
                                            ratios=((1 - validation_split), 0.0, validation_split),
                                            sliding_window = sliding_window,
                                            silent = False,
                                            discard_overlaps = False)

    delta = (sliding_window - 1) // 2
    train_idxs = train_idxs[np.where((train_idxs >= 2 * delta) & (train_idxs <= len(X['codes']) - 2 * delta))]
    val_idxs = val_idxs[np.where((val_idxs >= 2 * delta) & (val_idxs <= len(X['codes']) - 2 * delta))]

    """ Create Generator """
    train_gen = SlidingGenerator(x = X, y = Y,
                                 sliding_window = sliding_window,
                                 indexes = {ln: train_idxs for ln in outputs + ['codes']},
                                 sliding_function = {**{yn: 'mean' for yn in outputs}, **{'codes': 'none'}},
                                 batch_size = batch_size,
                                 shuffle = True)

    val_gen = SlidingGenerator(x = X, y = Y,
                               sliding_window = sliding_window,
                               indexes = {ln: val_idxs for ln in outputs + ['codes']},
                               sliding_function = {**{yn: 'mean' for yn in outputs}, **{'codes': 'none'}},
                               batch_size = batch_size,
                               shuffle = False)


    full_gen = None


    return train_gen, val_gen, full_gen, input_shape, output_shape, {'train': train_idxs, 'validation': val_idxs}

""" Prepare Regression data """
def prepare_reg_data(data_norm, codes, smooth_window = 121, sliding_window = 21, batch_size = 32,
                     validation_split = .1, blocksize = 21):

    """ Define inputs (also outputs) """
    outputs = ['PERM','PHIE']

    """ Get number of samples """
    delta = (sliding_window - 1) // 2

    """ Crop codes """
    codes = codes[delta:]

    """ Smooth codes """
    codes_cropped = np.concatenate(
        [np.convolve(codes[:, i], np.ones((smooth_window,)) / smooth_window, mode='same')[:, None] for i in
         range(codes.shape[1])], axis=1)

    """ Crop data """
    X = {'codes': codes_cropped}
    Y = {yn: data_norm[yn][delta:-delta] for yn in outputs}

    """ Calculate data_shape """
    input_shape = {xn: X[xn].shape[1:] for xn in X}
    output_shape = {yn: Y[yn].shape[1:] for yn in Y}

    """ Define indexes """
    train_idxs, _, val_idxs = divide_blocks(Z = X['codes'],
                                            blocksize = blocksize,
                                            ratios=((1 - validation_split), 0.0, validation_split),
                                            sliding_window = sliding_window,
                                            silent = False,
                                            discard_overlaps = False)

    delta = (sliding_window - 1) // 2
    train_idxs = train_idxs[np.where((train_idxs >= 2 * delta) & (train_idxs <= len(X['codes']) - 2 * delta))]
    val_idxs = val_idxs[np.where((val_idxs >= 2 * delta) & (val_idxs <= len(X['codes']) - 2 * delta))]

    """ Create Generator """
    train_gen = SlidingGenerator(x = X, y = Y,
                                 sliding_window = sliding_window,
                                 indexes = {ln: train_idxs for ln in outputs + ['codes']},
                                 sliding_function = {**{yn: 'mean' for yn in outputs}, **{'codes': 'none'}},
                                 batch_size = batch_size,
                                 shuffle = True)

    val_gen = SlidingGenerator(x = X, y = Y,
                               sliding_window = sliding_window,
                               indexes = {ln: val_idxs for ln in outputs + ['codes']},
                               sliding_function = {**{yn: 'mean' for yn in outputs}, **{'codes': 'none'}},
                               batch_size = batch_size,
                               shuffle = False)

    full_gen = SlidingGenerator(x=X, y=Y,
                               sliding_window=sliding_window,
                               indexes={ln: np.arange(delta,codes_cropped.__len__()-delta) for ln in outputs + ['codes']},
                               sliding_function={**{yn: 'mean' for yn in outputs}, **{'codes': 'none'}},
                               batch_size=batch_size,
                               shuffle=False)


    return train_gen, val_gen, full_gen, input_shape, output_shape, {'train': train_idxs, 'validation': val_idxs}

""" Prepare Regression data """
def prepare_Class_data2(data_norm, smooth_window = 121, sliding_window = 21, batch_size = 32,
                     validation_split = .1, blocksize = 21):
    print("****************** Prepare Classication Data SlidingGenerator .....")
    """ Define inputs (also outputs) """
    inputs = ['Acoustic']
    outputs = ['FDISSOL']

    """ Get number of samples """
    delta = (sliding_window - 1) // 2


    """ Crop data """
    X = {yn: data_norm[yn][delta:-delta] for yn in inputs}
    Y = {yn: data_norm[yn][delta:-delta] for yn in outputs}
    print(X)

    """ Calculate data_shape """
    input_shape = {xn: X[xn].shape[1:] for xn in X}
    output_shape = {yn: Y[yn].shape[1:] for yn in Y}

    """ Define indexes """
    train_idxs, _, val_idxs = divide_blocks(Z = X['Acoustic'],
                                            blocksize = blocksize,
                                            ratios=((1 - validation_split), 0.0, validation_split),
                                            sliding_window = sliding_window,
                                            silent = False,
                                            discard_overlaps = False)

    delta = (sliding_window - 1) // 2
    train_idxs = train_idxs[np.where((train_idxs >= 2 * delta) & (train_idxs <= len(X['Acoustic']) - 2 * delta))]
    val_idxs = val_idxs[np.where((val_idxs >= 2 * delta) & (val_idxs <= len(X['Acoustic']) - 2 * delta))]

    """ Create Generator """
    train_gen = SlidingGenerator(x = X, y = Y,
                                 sliding_window = sliding_window,
                                 indexes = {ln: train_idxs for ln in data_norm},
                                 sliding_function = {**{yn: 'mean' for yn in outputs}, **{'Acoustic': 'none'}},
                                 batch_size = batch_size,
                                 shuffle = True)

    val_gen = SlidingGenerator(x = X, y = Y,
                               sliding_window = sliding_window,
                               indexes = {ln: val_idxs for ln in data_norm},
                               sliding_function = {**{yn: 'mean' for yn in outputs}, **{'Acoustic': 'none'}},
                               batch_size = batch_size,
                               shuffle = False)




    return train_gen, val_gen, input_shape, output_shape, {'train': train_idxs, 'validation': val_idxs}


def load_dataClassNewMarcio(data_dirIMG=None, data_dirClass=None, data_dirSave=None, MakeChannels=None, dataPred=True, DataTrain=None, sep=None, LabelWell="W0", Blind=None):
    # inputs : CSVs
    # outputs : Dict with : Array Data acoustic normalize 0-1 float32 (U) and array labels categorical FDISSOL - int32
    
    print("Folder actual: " + data_dirSave)
    fileImg = data_dirIMG[:-4]
    fileClass = data_dirClass[:-4]


#     print("############################################################################################")
#     print("#### LOAD BOREHOLE: " + fileImg)
#     print("############################################################################################")

    filePKL = os.path.join(data_dirSave, "data" + LabelWell + ".pkl")
    filePKLNorm = os.path.join(data_dirSave, "dataNorm" + LabelWell + ".pkl")
    if os.path.isfile(filePKL) and os.path.isfile(filePKLNorm):
        print("Load files Dataframe in " + filePKLNorm)
        data_norm = load_resultsPKL(path = filePKLNorm)
        data = load_resultsPKL(path = filePKL)
#         data_norm = pd.DataFrame(data_norm)
#         data = pd.DataFrame(data)
        
        
    else:
        
        if os.path.isfile(fileImg + '.npy') and \
                os.path.isfile(fileClass + '.npy') and \
                os.path.isfile(fileClass +  '_Depth' + '.npy'):
            print("Exist : " + fileImg + '.npy')
            print("Exist : " + fileClass + '.npy')
            print("Exist : " + fileClass +  '_Depth' + '.npy')

            """ Load data """
            U = np.load(fileImg + '.npy')
            FDISSOL = np.load(fileClass + '.npy')
            Depth = np.load(fileClass +  '_Depth' + '.npy')
            print("Data Shape:")
            print("U", U.shape)
            print("Class", FDISSOL.shape)
            print("Depth" , Depth.shape)
            """ Build data """
            data = {'Acoustic': UT,
                         'FDISSOL': FDISSOL,
                         'Depth': DepthAMP
                         }

            data_norm = {'Acoustic': U,
                    'FDISSOL': FDISSOL,
                    'Depth': DepthAMP
                    }

            save_resultsPKL(data_norm, path = os.path.join(data_dirSave, "dataNorm.pkl"))
            save_resultsPKL(data, path = os.path.join(data_dirSave, "data.pkl"))
        else:
            ##############
            ##### condição pra verificar se precisa interpo;=lacao
            ##############
            CatIMGv = pd.read_csv(data_dirIMG, delimiter=",") 
            CatIMGpv = pd.read_csv(data_dirIMG, delimiter=";")
            if CatIMGv.shape[1] > 1:
                CatIMG = CatIMGv
            else:
                CatIMG = CatIMGpv
            del CatIMGv
            del CatIMGpv
            print("Loading data Images in: " + data_dirIMG)
#                 print(data_dirIMG[i])
#                 print(CatIMG)
            CatIMGA = np.array(CatIMG.values)
            if not Blind:
                CatClassv = pd.read_csv(data_dirClass, delimiter=",") 
    #                 print(CatClassv.shape)
                CatClasspv = pd.read_csv(data_dirClass, delimiter=";") 
    #                 print(CatClasspv.shape)
                if CatClassv.shape[1] > 1:
                    CatClass = CatClassv
                else:
                    CatClass = CatClasspv
                del CatClassv
                del CatClasspv

                print("Loading data Classes in: " + data_dirClass)
    #                 print(data_dirClass[i])
    #                 print(CatClass)

            else:
                CatClass = np.zeros((CatIMGA.shape[0],2))
            U = CatIMGA[1:, 1:-1]

            if U.shape[1] > 180:
                U = CatIMGA[1:, 1:1+180]

            if U.shape[1] < 180:
                U = CatIMGA[1:, 1:]

            U = np.float64(U)
            ic = np.float64(-100.00)
#             U = np.where(U <= ic, np.max(U), U)

            CatClassA = np.array(CatClass.values)

            DepthAMP = CatIMGA[1:, 0]


            FDISSOL = CatClassA[1:, 1]
            Depth = CatClassA[1:, 0]

            del CatClassA

            print("Data Shape BEFORE interpolation:")
            print("U", U.shape)
            print("Depth AMP", DepthAMP.shape)
            print("Class", FDISSOL.shape)
            print("Depth Dissol", Depth.shape)

            ####### INTERPOLATION
            PETRO = {"Depth": Depth, "FDISSOL": FDISSOL}
            """ Now crop depth between in common interval """
            d0 = CatIMGA[1:, 0] # depth data big
            d1 = PETRO["Depth"]
            for v in d0:
                if isinstance(v, str):
#                     print(type(v))
#                     print(v)
                    d0 = np.float64(CatIMGA[1:, 0]) # depth data big
                    d1 = np.float64(PETRO["Depth"])
                    break
            if not Blind:
                # d2 = FDISSOL
                dres0 = np.min(np.diff(np.sort(d0)))
                dres1 = np.min(np.diff(np.sort(d1)))

                dres = np.minimum(dres0, dres1)
                dmin = np.maximum(d0.min(), d1.min())
                dmax = np.minimum(d0.max(), d1.max())

                """ Create new depth """
                dref = np.arange(dmin, dmax, dres)

                """ Now crop new data """
                hh = []
                for j in d1:
                    if j >= dmin or j <= dmax:
                        hh.append(j)

                PETRO["Depth"] = hh


                """ Crop old data too """
                idxs2keep = (d0 >= dmin) & (d0 <= dmax)
                d0 = d0[idxs2keep]
                U = U[idxs2keep, :]
                DepthAMP = d0

                """ Interpolate new data """
                fP = interpolate.interp1d(PETRO['Depth'], PETRO["FDISSOL"])
                FD = fP(dref)
                FDISSOL = FD

            if U.shape[0] != FDISSOL.shape[0]:
                print("Error")
                assert U.shape[0] != FDISSOL.shape[0], "Error Invalid Operation shape Image no match shape classifications"

            if U.shape[1] != 180:
                assert U.shape[1] != 180, "Error Invalid Operation shape Image no match shape 180 columns"

          
            """ Define float type """

            U = np.float32(U)
            FDISSOL = np.uint8(FDISSOL)


            if MakeChannels:
       
                UO = U.copy()
                UOV = np.ravel(UO)
                MIN = np.min([i for i in UOV if (i > ic and not math.isnan(i))])
                MAX = np.max([i for i in UOV if (i > ic and not math.isnan(i))])

                UON = (UO - MIN) / (MIN + MAX)
#                 print(MAX, MIN)

                Ulog = np.log10(U)
                UlogV = np.ravel(Ulog)
                MIN = np.min([i for i in UlogV if (i > ic and not math.isnan(i))])
                MAX = np.max([i for i in UlogV if (i > ic and not math.isnan(i))])

                UlogN = (Ulog - MIN) / (MIN + MAX)
                print("Max and Min log images")
                print(MAX, MIN)

                UG = ndimage.gaussian_gradient_magnitude(U, sigma=1)
                UGV = np.ravel(UG)
                MIN = np.min([i for i in UGV if (i > ic and not math.isnan(i))])
                MAX = np.max([i for i in UGV if (i > ic and not math.isnan(i))])

                UGN = (UG - MIN) / (MIN + MAX)
                print("Max and Min gauss images")
                
                print(MAX, MIN)

                UT = np.zeros((UO.shape[0], UO.shape[1], 3))
                UT[:, :, 0] = np.nan_to_num(UO)
                UT[:, :, 1] = np.nan_to_num(Ulog)
                UT[:, :, 2] = np.nan_to_num(UG)
                # UT = np.where(UT<=-900, 1, UT)

                print(UT.max(), UT.min())


                UTN = np.zeros((UO.shape[0], UO.shape[1], 3))
                UTN[:, :, 0] = UON
                UTN[:, :, 1] = UlogN
                UTN[:, :, 2] = UGN
                print(UTN.max(), UTN.min())
                U = UTN.copy()
            else:
                """ Normalize images """
                UT = U
                MIN = np.min([i for i in np.ravel(U) if (i > ic and not math.isnan(i))])
                MAX = np.max([i for i in np.ravel(U) if (i > ic and not math.isnan(i))])
                U = (U - MIN) / (MIN + MAX)
                U = U[:, :, None]

            F = []
            for i in FDISSOL:
                F.append(str(i))

            FDISSOL = F

#             print(U.shape)
#             print(len(FDISSOL))

            print("Data Shape AFTER interpolation:")
            print("U", U.shape)
            print("U max no norm", UT.max())
            print("U max no norm", UT.min())
            print("U max norm", U.max())
            print("U max norm", U.min())
            print("Depth AMP", DepthAMP.shape)
            print("Class", len(FDISSOL))
            print("Depth Dissol", Depth.shape)

            """ Build data """
            data = {'Acoustic': UT,
                         'FDISSOL': FDISSOL,
                         'Depth': DepthAMP
                         }

            data_norm = {'Acoustic': U,
                    'FDISSOL': FDISSOL,
                    'Depth': DepthAMP
                    }
            
            """ Save data """

            save_resultsPKL(data_norm, path = filePKLNorm)
            save_resultsPKL(data, path = filePKL)
#     else:
#         """ Normalize  """
        # coeffs = None #{xn: [np.nanmin(data[xn]), np.nanmax(data[xn])] for xn in data}
        # data_norm = None #{yn: (data[yn] - coeffs[yn][0]) / (coeffs[yn][1] - coeffs[yn][0]) for yn in data}
#         data_norm = None
#         data = None
#         data_norm = load_resultsPKL(path = os.path.join(data_dirSave,"dataNorm.pkl"))
#         data = load_resultsPKL(path = os.path.join(data_dirSave,"data.pkl"))
#         print(data_norm)

    return data_norm, data
