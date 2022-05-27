#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:38:39 2018

@author: manu
"""
from __future__ import print_function
import numpy as np
import numpy.linalg as linalg
import random, time, os, string
import matplotlib
import bisect # for image histogram normalization
from bisect import bisect_left
from sklearn.model_selection import StratifiedKFold

# for datetime stamp conversions
from time import gmtime, strftime
import re

# Interactive windows
# import Tkinter, tkFileDialog
# from Tkinter import *
# import Tkinter as ttk
# from ttk import *

# Make sure that we are using QT5
# matplotlib.use('Qt5Agg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtCore import QCoreApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import sys
from sys import exit

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# app = QCoreApplication.instance()
# if app is None:
#     app = QApplication(sys.argv)


# from tensorflow.keras.utils.vis_utils import model_to_dot # from tensorflow.keras.utils import plot_model
# from keras.layers import Wrapper

import PIL
from PIL import Image, ImageFont, ImageDraw

#
# >> normalize([1, 2, 3, 4, 5, 1, 4, 1, 2, 4, 5])
#
def normalize(X):
    
    if type(X) == np.ndarray:
        minX = X.min()
        maxX = X.max()
        
        return (1.0*X-minX)/(maxX-minX)
    elif type(X) == list:
        minX = min(X)
        maxX = max(X)
        
        return [(1.0*X[i]-minX)/(maxX-minX) for i in range(len(X))]
    else:
        raise ValueError('Unsupported data type: %s' % (type(X)))


# This function takes an input src and returns a normalized version [0,255] whose histogram 
# has been stretched to occupy the greatest range possible. This function helps increasing 
# the contrast of the image
def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r,c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    if (vin[1]-vin[0]) > 0:
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
        for r in range(dst.shape[0]):
            for c in range(dst.shape[1]):
                vs = max(src[r,c] - vin[0], 0)
                vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
                dst[r,c] = vd
    return dst   


# This function is useful to get the unique values of a list of strings, as well
# as to obtain the indexes of each unique value. 
# example: 
#
# >> l = ['foo','faa','fee','fii','fuu','fae','fao','fai','faa','foo','fuu']
# >> u,i = uniqueStr(l)
# >> print(u)
#    ['fii', 'fai', 'fee', 'foo', 'fuu', 'fao', 'faa', 'fae']
# >> print(i)
#    [3, 6, 2, 0, 4, 7, 5, 1, 6, 3, 4]
def uniqueStr(l):
    u = list(set(l))
    i = []
    for c in l:
        i.append(u.index(c))
        
    return u,i


# This function is useful for finding which elements of "a" are present in "b".
#
# >> ismember([1,2,3,4],[1,5,2])
#    [True, True, False, False]
#
def ismember(a, b):
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = True
        return [bind.get(itm, False) for itm in a]  # None can be replaced by any other "not in b" value


# Check if directory exists, in case it does not, create it
def mkdir(dd):
    try:
        os.stat(dd)
    except:
        os.mkdir(dd)    


# This function acts the same way as any train/test/validation division fcn
# but instead of dividing the data into single point samples, it divides it
# into continuous blocks with size "blocksize". Q is the total number of 
# samples, blocksize is the size of the blocks and the train, val and test
# ratios are the ratios of each subsample that will be used for training, 
# validating and testing the net or the model we want to use. Overlapsize
# guarantees that, if we are using a sliding window to iterate through the
# data, no overlapping indexes are used (that is, some indexes will be dropped
# so that no indexes are used for training/testing/validating)
#
# >> trainIdxs, valIdxs, testIdxs = divideblocksize(1000,10,0.8,0,0.2,overlapsize=1,display=False)
# >>
#


def divideblocksizeTensor(X, Y, ws, train_split, test_split, validation_split, LabelBH, reloadIDX=True):
    plotconf = True
    tt = np.arange(0,len(Y),ws)
#     print(X.shape)
    if tt[-1] == len(Y):
        X = X
        Y = Y 
    else:
        X = X[0:tt[-1]]
        Y = Y[0:tt[-1]]

    if len(X.shape) > 2:
        X0 = np.zeros((X.shape[0],ws,X.shape[1], X.shape[2]))
    else:
        X0 = np.zeros((X.shape[0],ws,X.shape[1]))
    # criando X tensor baseado em Y
    for n,y in enumerate(Y):
        b = list(Y[n:n+ws])
        c = Y[n]
        d = np.array([i == c for i in b])
        if X0[n].shape[0] <= X[n:n+ws].shape[0]:
            e = X[n:n+ws]   
            if d.all():
    #             print(X0[n].shape)
    #             print(e.shape)
                X0[n] = e
    #             print("Iguais")
    #             print(X0[n].shape)
                if plotconf and not os.path.isfile("TestImage" + "WS" + str(ws) + "_comYtodoIgual.png"):
                    plt.figure()
                    plt.imshow(e, cmap="gray")
                    plt.plot(np.zeros((n+ws,)), np.arange(0,len(b),1), "C" + str(c), linewidth=20)
                    plt.savefig("TestImage" + "WS" + str(ws) + "_comYtodoIgual.png")
            else:
                xw = np.zeros(e.shape)
                if plotconf and not os.path.isfile("TestImage" + "WS" + str(ws) + "_comYdiferente.png"):
                    plt.figure()
                    plt.imshow(e, cmap="gray")
                
                for m,i in enumerate(Y[n:n+ws]):
                    if plotconf and not os.path.isfile("TestImage" + "WS" + str(ws) + "_comYdiferente.png"):
                        plt.plot(0.0, m, color="C" + str(i), marker="s", markersize=20)
                    
                    xx = 0
                    if i == Y[n]:
                        xw[m] = X[n + m]
                    else:
                        xw[m] = X[n]
                X0[n] = xw
                if plotconf and not os.path.isfile("TestImage" + "WS" + str(ws) + "_comYdiferente.png"):
                    plt.savefig("TestImage" + "WS" + str(ws) + "_comYdiferente.png")
                    plt.figure()
                    plt.imshow(xw, cmap="gray")
                    plt.plot(np.zeros((len(b),)), np.arange(0,len(b),1), "C" + str(Y[n]), linewidth=20)
                    plt.savefig("TestImage" + "WS" + str(ws) + "_comYdiferenteCORRIGIDO.png")
             
    X = np.array(X0)
    Y = np.array(Y)
    import pickle
         
    N = X.shape[0]
    if reloadIDX:
        ids = {"ids" : np.random.permutation(N)}
        with open("Index" + LabelBH + ".pkl",'wb') as f:
            pickle.dump(ids,f)
    else:
        if os.path.exists("Index" + LabelBH + ".pkl"):
            with open("Index" + LabelBH + ".pkl",'rb') as f:
                ids = pickle.load(f)
        else:
            ids = {"ids" : np.random.permutation(N)}
            with open("Index" + LabelBH + ".pkl",'wb') as f:
                pickle.dump(ids,f)
    ids = ids["ids"]
    print(ids)
    
    ntrain = int(N*validation_split)
    print(ntrain)
    ntrain2 = int(N*test_split)
    print(ntrain2)
    Xtrain = X[ids[ntrain + ntrain2:]]
    Ytrain = Y[ids[ntrain + ntrain2:]]
    Xtest = X[ids[ntrain:ntrain+ntrain2]]
    Ytest = Y[ids[ntrain:ntrain+ntrain2]]
    Xval = X[ids[:ntrain]]
    Yval = Y[ids[:ntrain]]
    ids = {"train": ids[ntrain + ntrain2:], "test": ids[ntrain:ntrain+ntrain2], "val": ids[:ntrain]}
            
    return Xtrain, Ytrain, Xtest, Ytest, Xval, Yval, ids

def load_data_kfold(k, X_train, Y_train):
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X_train, Y_train))


    return folds


def divideblocksizeTensor_ToKfold(X, Y, ws, k, train_split, test_split, validation_split, reloadIDX=False):
    plotconf = True
    tt = np.arange(0,len(Y),ws)
#     print(X.shape)
    if tt[-1] == len(Y):
        X = X
        Y = Y 
    else:
        X = X[0:tt[-1]]
        Y = Y[0:tt[-1]]

    if len(X.shape) > 2:
        X0 = np.zeros((X.shape[0],ws,X.shape[1], X.shape[2]))
    else:
        X0 = np.zeros((X.shape[0],ws,X.shape[1]))
    # criando X tensor baseado em Y
    for n,y in enumerate(Y):
        b = list(Y[n:n+ws])
        c = Y[n]
        d = np.array([i == c for i in b])
        if X0[n].shape[0] <= X[n:n+ws].shape[0]:
            e = X[n:n+ws]   
            if d.all():
    #             print(X0[n].shape)
    #             print(e.shape)
                X0[n] = e
    #             print("Iguais")
    #             print(X0[n].shape)
                if plotconf and not os.path.isfile("TestImage" + "WS" + str(ws) + "_comYtodoIgual.png"):
                    plt.figure()
                    plt.imshow(e, cmap="gray")
                    plt.plot(np.zeros((n+ws,)), np.arange(0,len(b),1), "C" + str(c), linewidth=20)
                    plt.savefig("TestImage" + "WS" + str(ws) + "_comYtodoIgual.png")
            else:
                xw = np.zeros(e.shape)
                if plotconf and not os.path.isfile("TestImage" + "WS" + str(ws) + "_comYdiferente.png"):
                    plt.figure()
                    plt.imshow(e, cmap="gray")
                
                for m,i in enumerate(Y[n:n+ws]):
                    if plotconf and not os.path.isfile("TestImage" + "WS" + str(ws) + "_comYdiferente.png"):
                        plt.plot(0.0, m, color="C" + str(i), marker="s", markersize=20)
                    
                    xx = 0
                    if i == Y[n]:
                        xw[m] = X[n + m]
                    else:
                        xw[m] = X[n]
                X0[n] = xw
                if plotconf and not os.path.isfile("TestImage" + "WS" + str(ws) + "_comYdiferente.png"):
                    plt.savefig("TestImage" + "WS" + str(ws) + "_comYdiferente.png")
                    plt.figure()
                    plt.imshow(xw, cmap="gray")
                    plt.plot(np.zeros((len(b),)), np.arange(0,len(b),1), "C" + str(Y[n]), linewidth=20)
                    plt.savefig("TestImage" + "WS" + str(ws) + "_comYdiferenteCORRIGIDO.png")
             
    X = np.array(X0)
    Y = np.array(Y)
    import pickle
         
    N = X.shape[0]
    if not os.path.isfile("Index.pkl"):
        if reloadIDX:
            ids = {"ids" : np.random.permutation(N)}
            with open("Index.pkl",'wb') as f:
                pickle.dump(ids,f)
    else:
        with open("Index.pkl",'rb') as f:
            ids = pickle.load(f)
    ids = ids["ids"]
            
    ntrain = int(N*validation_split)
#     ntrain2 = int(N*test_split)
    
    Xtrain = X[ids[ntrain:]]
    Ytrain = Y[ids[ntrain:]]
    
    Xval = X[ids[:ntrain]]
    Yval = Y[ids[:ntrain]]
    ids = {"trainCV": ids[ntrain:], "val": ids[:ntrain]}
            
    return Xtrain, Ytrain, Xval, Yval, ids


def divideblocksize(Q,blocksize,trainRatio,valRatio,testRatio,overlapsize=1,display=False,silent=False):

    # First we must check if all ratios sum 1, in case they don't take the 
    # one with the minimum value and change it
    if ((trainRatio + valRatio + testRatio) > 1) or ((trainRatio + valRatio + testRatio) < 0):
        raise ValueError('The ratios of the specified indexes MUST sum up to 1.')
    
    # Now let's check if we can actually avoid overlapping of blocks
    nseeds = (Q*(testRatio+valRatio)//blocksize) # number of seeds that we will use to initialize the blocks
    ntestval = nseeds*(blocksize + 2*overlapsize)

    if (nseeds <= 0):
        raise ValueError('The specified blocksize is too big or testing/validation proportions are too small to be able to guarantee non-overlapping blocks.')

    # Now that we can assure non-overlapping regions, let's start an iteration
    # process to divide the seeds over the total number of samples Q
    seeds = [] 
    potSeeds = np.arange(0,Q-blocksize) # potential seeds (remaining seeds)
    seeds2discard = []
    while np.shape(seeds)[0] < nseeds*blocksize:
        # take one random seed
        seedtmp = potSeeds[random.randint(0,len(potSeeds)-1)]
        seedtmp = np.arange(seedtmp,seedtmp+blocksize)
        seeds.extend(seedtmp)
        
        # Now remove from potSeeds all seeds in seedtmp
        seeds2remove = np.arange(seedtmp[0]-overlapsize,seedtmp[0]+blocksize+overlapsize)
        seeds2discard.extend(seeds2remove)
        seeds2keep = ismember(potSeeds,seeds2remove)
        seeds2keep =  [not i for i in seeds2keep]
        potSeeds = potSeeds[seeds2keep]
        if not silent:
            print("Seed set @ %i... #seeds = %i" % (seedtmp[0],np.shape(seeds)[0]))
    
    
    # Now let's divide the seeds between testing and validating, according to
    # the proportion specified by the user
    Qval = int(np.floor(valRatio*Q))
    
    valIdxs = seeds[0:Qval]
    testIdxs = seeds[Qval:]
    
    # Now the train indexes are the ones that are left in the potSeeds
    trainIdxs = np.arange(0,Q-overlapsize)
    trainIdxs = trainIdxs[[not i for i in ismember(trainIdxs,seeds2discard)]]
    
    # Now we can plot the indexes using an RGB image
    if display:
        
        W = 100
        RGB = 255*np.ones((Q,W,3),dtype=np.uint8)
        # green for trainIdxs
        # blue for valIdxs
        # red for testIdxs
        R = np.concatenate((255*np.ones((Q,W,1),dtype=np.uint8),np.zeros((Q,W,2),dtype=np.uint8)),axis=2)
        G = np.concatenate((np.zeros((Q,W,1),dtype=np.uint8),255*np.ones((Q,W,1),dtype=np.uint8),np.zeros((Q,W,1),dtype=np.uint8)),axis=2)
        B = np.concatenate((np.zeros((Q,W,2),dtype=np.uint8),255*np.ones((Q,W,1),dtype=np.uint8)),axis=2)
        RGB[trainIdxs,:,:] = G[trainIdxs,:,:]
        RGB[testIdxs,:,:] = R[testIdxs,:,:]
        RGB[valIdxs,:,:] = B[valIdxs,:,:]
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(W/20,Q/20))
        axes.imshow(RGB)
        # pass the figure to the custom window
        a = ScrollableWindow(fig)

    return trainIdxs,valIdxs,testIdxs


# Print iterations progress
def printProgressBar (t0, iteration, total, delta=None, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        delta       - Optional  : space (in percentage) between prints
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    
    if (delta is None):
        delta = 1
    else:
        delta = np.floor(delta/100*total)
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total-1)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    
    # Print New Line on Complete
    if iteration == total-1: 
        suffix = '(' + ElapsedTime(t0) + ' elapsed)' + suffix
        sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
        sys.stdout.flush()
        #print()
    else:
        # only print if mod (iteration % delta is zero)
        if (iteration % delta) == 0:
            suffix = '(' + TicTacTimer(t0,(1.0*iteration)/total) + ' left)' + suffix
            sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
            sys.stdout.flush()

def OpenFileDialog(title='Select your data',filetypes = (("csv files","*.csv"),("all files","*.*"))):
    root = Tkinter.Tk()
    root.withdraw()
    f = tkFileDialog.askopenfilename(title=title,filetypes=filetypes)
    return f

def OpenFolderDialog(title="Select folder"):
    root = Tkinter.Tk()
    root.withdraw()
    d = tkFileDialog.askdirectory(parent=root,title=title)
    return d

def SelectChoicesIndex(prefix='Select Index: ',choices=None):
    sel_id = -1
    if (choices is not None):
        while (sel_id < 0) or (sel_id > len(choices)-1):
            print(prefix)
            [print('\t' + str(i) + ' -> ' + choices[i] + '')  for i in np.arange(0,len(choices))]
            sel_id = input(('Selected index [0 to %d] : ') % (len(choices)-1))
            
            if (sel_id < 0) or (sel_id > len(choices)-1):
                print('\n\nWrong index. Please insert a valid number between 0 and %d.\n' % (len(choices)-1))
                sel_id = -1
    return sel_id

# This function is useful to estimate the time left for a certain process with "p" progress ([0,1]) to finish. 
# t0 represents the starting time when the iterative process was first called.
def TicTacTimer(t0,p):
    tnow = time.time()
    hh = 0
    mm = 0
    ss = 0
    
    if p > 0:
        tf = (tnow-t0)/p
        tf = tf-(tnow-t0)
        hh = tf//3600
        tf = tf-(hh*3600)
        mm = tf//60
        ss = tf - (mm*60)
    
    return "%02d:%02d:%02d" % (hh,mm,ss)

def ElapsedTime(t0):
    tnow = time.time()
    hh = 0
    mm = 0
    ss = 0
    tep = tnow-t0
    hh = tep//3600
    tep = tep - (hh*3600)
    mm = tep//60
    ss = tep - (mm*60)
    
    return "%02d:%02d:%02d" % (hh,mm,ss)

def getTerminalSize():
    import os
    env = os.environ
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct, os
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
        '1234'))
        except:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

        ### Use get(key[, default]) instead of a try/catch
        #try:
        #    cr = (env['LINES'], env['COLUMNS'])
        #except:
        #    cr = (25, 80)
    return int(cr[1]), int(cr[0])

def string2date(s=None):
    if not s:
        s = strftime("%Y%m%d%H%M%S",gmtime())
    return ('%s/%s/%s %s:%s:%s' % (s[6:8],s[4:6],s[0:4],s[8:10],s[10:12],s[12:15]))

def yesnoquestion(question=None):
    answer = None
    while answer not in ['y','n']:
        answer = raw_input(question + " [y/n]: ")
    
    return answer == 'y'

def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return 0
        #return myList[0]
    if pos == len(myList):
        return len(myList)
        #return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return pos
    else:
       return pos-1

def getvalidanswer(question=None,answertype=str,invalid_set=set(string.punctuation),nonEmpty=False):
    askagain = True
    if answertype==str:
        answertype=int
    while askagain:
        answer = raw_input(question)
        if nonEmpty:
            if (answer == ''):
                askagain = True
                print('WARNING: you entered an empty answer but this field cannot be empty. Please enter a valid answer.')
                continue
        try:
            answertype(answer) # it's a number, so ask again
            askagain = True
            print('WARNING: You entered an invalid channel name. Please use only character-starting channel names. Do not start your channel name with a number and do not use special characters either (' + ', '.join(invalid_set) + ').')
        except:
            if (answer == ''):
                askagain = False
            else:
                # check if channel name has special characters
                if any(char in invalid_set for char in answer):
                    # invalid filename
                    askagain = True
                    print('WARNING: You entered an invalid channel name. Please use only character-starting channel names. Do not start your channel name with a number and do not use special characters either (' + ', '.join(invalid_set) + ').')
                else:
                    askagain = False
    return answer

def tupleDepth(l):
    depths = [tupleDepth(item) for item in l if isinstance(item, tuple)]

    if len(depths) > 0:
        return 1 + max(depths)

    return 1


"""Helper to handle indices and logical indices of NaNs.

Input:
    - y, 1d numpy array with possible NaNs
Output:
    - nans, logical indices of NaNs
    - index, a function, with signature indices= index(logical_indices),
      to convert logical indices of NaNs to 'equivalent' indices
Example:
    >>> # linear interpolation of NaNs
    >>> nans, x= nan_helper(y)
    >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
"""
def nan_helper(x):
    return np.isnan(x), lambda z: z.nonzero()[0]

def find_ellipse(x, y):
    
    def fitEllipse(x,y):
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  linalg.eig(np.dot(linalg.inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:,n]
        return a
    
    def ellipse_center(a):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
        return np.array([x0,y0])
    
    def ellipse_angle_of_rotation( a ):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        return 0.5*np.arctan(2*b/(a-c))
    
    def ellipse_axis_length( a ):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)
        return np.array([res1, res2])
    
    xmean = x.mean()
    ymean = y.mean()
    x -= xmean
    y -= ymean
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    center[0] += xmean
    center[1] += ymean
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    x += xmean
    y += ymean
    return center, phi, axes

def allmaps():
    return ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']


def findChangeIntervals(V,min_change=1):
    
    if V is None:
        return None,None
    
    N = len(V)
    if N <= 1:
        return 1,V
    
    if type(V[0]) is bool:
        P = [1.*v for v in V]
    elif type(V[0]) is string:
        raise ValueError('This has not been implemented yet')
    
    dP = np.abs(np.diff(P))
    iP = [i for i,dpi in enumerate(dP) if dpi >= min_change]
    
    if iP == []:
        iP = [1,N]
    else:
        iP = [-1] + iP + [N-1]
        
    iV = []
    for i in range(len(iP)-1):
        iV.append([iP[i]+1,iP[i+1]+1])
        
    muP = []
    for i in iV:
        muP.append(np.mean(P[i[0]:i[1]]))
    
    if type(V[0]) is bool:
        muP = [v == 1 for v in muP]
    
    return iV,muP


""" Confuision matrix analysis. Plots confusion table with percentages and returns it
"""
def cm_analysis(y_true=None, y_pred=None, filename="", classes=None, labels=None, normalize = False, ymap=None, figsize=(10,10), cmap='Blues'):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap != None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]


    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm_perc, index=classes, columns=classes)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap=cmap, annot_kws = {"size":26})
    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    if filename is not "" and filename is not None:
        plt.savefig(filename)

    return cm   

    

""" This function is used to plot any keras model, and it works as an alternative
    to the builtin functions that come with keras such as "plot_model" or "model_to_dot".
    Its use is pretty straightforward: 
        
        0) Define your model using keras: 
            from keras.layers import Input,Dense,Conv2D,Flatten
            from keras.models import Model
            
            num_outputs = 1
            input_shape = (100,100,1)
            
            input = Input(shape=input_shape)
            layer = Conv2D(32,(3,3),activation='relu')(input)
            layer = Flatten()(layer)
            layer = Dense(units=64,activation='relu')(layer)
            layer = Dense(units=num_outputs,activation='softmax')(layer)
            
            model = Model(input,layer)
            
        1) Now call the plot function to visualize your model using:
            myModelRGB = plot_nice_model(model,display=True)
            
        2) Finally, you can store your model into an image file using opencv:
            import cv2
            cv2.imwrite('myModel.png',cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR)) # It is necessary to use cvtColor function because opencv swaps the color channels.

    There are some parameters the user can set:
        plot_nice_model(
                -model -> Required parameter. It contains a keras model (keras.models.Model)
                -font -> Expected a string or a PIL.ImageFont.FreeTypeFont. Example: font="UbuntuMono-R.ttf"
                -fontsize -> Expected integer or double. Default value is 210 (the greater, the better quality the final graph will have.)
                -corner_radius -> Expected integer or double. Controls the radius of the boxes of each layers' tag. 
                -border -> Expected integer. Sets the size of the border of each layer's tag.
                -display -> Expected a boolean (True or False). If set to True (or unset) the final graph of the model will be shown in the console (or figure).
        )

"""
def plot_nice_model(model,font=None,fontsize=210,corner_radius=80,border=20,display=True):

    def _get_scale(ddot,layerIds,tagSizes):
        nlayers = len(layerIds)
        # Let's first calculate the scale as the largest scale possible for all tags
        scales = np.zeros((nlayers,))
        i = 1
        ddot_tmp = ddot[i]
        while (ddot_tmp != "stop"):
            ddot_type = ddot_tmp[0:4]
            
            # Regular expressions were built using the online tool: https://regex101.com/r/9y9n85/1       
            if ddot_type == "node":
                pattern = re.compile("node (\d+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) (?:\")(\w+)(?:\:) (\w+)(?:\") (\w+) (\w+) (\w+) (\w+)")
                matches = pattern.findall(ddot_tmp)[0]
                dotId = matches[0]
                #dotXc = matches[1]
                #dotYc = matches[2]
                dotW = matches[3]
                #dotH = matches[4]
                #dotLayerName = matches[5]
                #dotLayertype = matches[6]
                
                # find right layer id:
                j = [ij for ij,ly in enumerate(layerIds) if ly == dotId]
                
                if (j != []):
                    j = j[0]
                    scales[j] = tagSizes[j][1]/float(dotW)
                   
            i += 1
            ddot_tmp = ddot[i]    
        
        scale = np.max(scales)
        
        return scale
    
    def _set_tags_positions(ddot,layerIds):
        
        nlayers = len(layerIds)
        
        # now let's go through each line and keep only the nodes (discard edges)
        tagsPos = np.zeros((nlayers,4),dtype='int') #we want the bounding box so let's store the [x0,y0,x1,y1]
        linesPos = [] # index_from, index_to 
        # we are going to assume that the information in the ddot string for each node is like:
        # "node node_ID x_center y_center width height "layer_name: LayerType" ..."
        
        i = 1
        ddot_tmp = ddot[i]
        while (ddot_tmp != "stop"):
            ddot_type = ddot_tmp[0:4]
            
            # Regular expressions were built using the online tool: https://regex101.com/r/9y9n85/1
            if ddot_type == "edge":
                pattern = re.compile("edge (\d+) (\d+) (\d+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) (\w+) (\w+)")
                try:
                    matches = pattern.findall(ddot_tmp)[0]
                    startId = matches[0]
                    endId = matches[1]
                    
                    _from = [ij for ij,ly in enumerate(layerIds) if ly == startId]
                    _to = [ij for ij,ly in enumerate(layerIds) if ly == endId]
                    
                    linesPos.append([_from[0],_to[0]])
                    
                    #linewidth = matches[2]
                    #dotlineX0 = matches[3]
                    #dotlineY0 = matches[4]
                    #dotlineX1 = matches[9]
                    #dotlineY1 = matches[10]
                        
                except:
                    print('edge exception, discarding.')
                
                
                
            elif ddot_type == "node":
                pattern = re.compile("node (\d+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) (?:\")(\w+)(?:\:) (\w+)(?:\") (\w+) (\w+) (\w+) (\w+)")
                matches = pattern.findall(ddot_tmp)[0]
                dotId = matches[0]
                dotXc = matches[1]
                dotYc = matches[2]
                #dotW = matches[3]
                #dotH = matches[4]
                #dotLayerName = matches[5]
                #dotLayertype = matches[6]
                
                # find right layer id:
                j = [ij for ij,ly in enumerate(layerIds) if ly == dotId]
                
                if (j != []):
                    j = j[0]
                    
                    tagsPos[j] = [int(float(dotXc)*scale) - tagSizes[j][1]//2,
                                  int(float(dotYc)*scale) + (tagSizes[j][0] - tagSizes[j][0]//2),
                                  int(float(dotXc)*scale) + (tagSizes[j][1] - tagSizes[j][1]//2),
                                  int(float(dotYc)*scale) - tagSizes[j][0]//2]
                
            i += 1
            ddot_tmp = ddot[i]  
            
        # Recalculate positions  
        offset_x = np.min((tagsPos[:,0],tagsPos[:,2]))
        tagsPos[:,0] -= offset_x
        tagsPos[:,2] -= offset_x
        
        offset_y = np.min((tagsPos[:,1],tagsPos[:,3]))
        tagsPos[:,1] -= offset_y
        tagsPos[:,3] -= offset_y
        
        # Let's now print each tag into their respective place in a RGB img
        _W = np.max([ts[2] for ts in tagsPos]) # Width of the image
        _H = np.max([ts[1] for ts in tagsPos]) # Height of the image
        
        # Invert Y positions
        tagsPos[:,3] = _H - tagsPos[:,3]
        tagsPos[:,1] = _H - tagsPos[:,1]
        
        offset_y = np.min((tagsPos[:,1],tagsPos[:,3]))
        tagsPos[:,1] -= offset_y
        tagsPos[:,3] -= offset_y
        
        _H = np.max([ts[3] for ts in tagsPos]) # Height of the image
            
        return tagsPos,linesPos,_W,_H
        
    
    
    def _get_layers_info(layers):
        layerTypes = [] # Variable to store the type of each layer (InputLayer, Conv2D, Concatenation, etc.)
        layerLabels = [] # Variable to store the label that we will print in the layer block, for each layer ("Input","Conv2D","Cat",etc.)
        layerIds = [] #Variable to store the index of each layer (so that we can identify them singularly) (0, 1, 2, etc.)
        
        for layer in layers:
            layer_id = str(id(layer))
        
            # Append a wrapped layer's label to node's label, if it exists.
            layer_name = layer.name
            class_name = layer.__class__.__name__
            if isinstance(layer, Wrapper):
                layer_name = '{}({})'.format(layer_name, layer.layer.name)
                child_class_name = layer.layer.__class__.__name__
                class_name = '{}({})'.format(class_name, child_class_name)
            if (class_name == 'Activation'):
                if (layer.activation.func_name == 'relu'):
                    class_name = 'ReLU'
            layerTypes.append(class_name)
            layerLabels.append(layer_name)
            layerIds.append(layer_id)
            
        return layerTypes, layerLabels, layerIds
    
    def _print_layers(layers,layerTypes,font,fontsize,top_margin,border,corner_radius):
            
        pad = 2
        tagSizes = []
        tagRGBs = []
        
        # now fill the conmap
        for i,(layer,ltype) in enumerate(zip(layers,layerTypes)):
            if (ltype == "InputLayer"):
                tag_color = "#acacace2"
                border_color = "#4d4d4da9"
                font_color = "#ffffffff"
                txt = "Input"
            elif (ltype == "Conv2D"):
                tag_color = "#2a7fffff"
                border_color = "#5151c0ff"
                font_color = "#ffffffff"
                txt = ltype
            elif (ltype == "MaxPooling2D"):
                tag_color = "#ff0000ff"    
                border_color = "#800000ff"
                font_color = "#ffffffff"
                txt = "MaxPool2D"
            elif (ltype == "AveragePooling2D"):
                tag_color = "#ff0000ff"    
                border_color = "#800000ff"
                font_color = "#ffffffff"
                txt = "AvgPool2D"
            elif (ltype == "BatchNormalization"):
                tag_color = "#ffff00ff"
                border_color = "#808000ac"
                font_color = "#000000ff"
                txt = "Bnorm"
            elif (ltype == "ReLU"):
                tag_color = "#00ff00ff"
                border_color = "#008000a9"
                font_color = "#000000ff"
                txt = ltype
            elif (ltype == "Flatten"):
                tag_color = "#cd8a63e2"
                border_color = "#a47a4aff"
                font_color = "#ffffffff"
                txt = ltype
            elif (ltype == "Dense"):
                tag_color = "#d42068e2"
                border_color = "#540023a9"
                font_color = "#ffffffff"
                txt = ltype + str(layer.units)
            elif (ltype == "Concatenate"):
                tag_color = "#ffffffff"
                border_color = "0000008f"
                font_color = "#000000ff"
                txt = "cat"
            else:
                tag_color = "#e9c6afff"
                border_color = "#005444a9"
                font_color = "#0000004f"
                txt = ltype
            
            tagSize = font.getsize(txt)
            
            
            H = 2*top_margin + 2*border + tagSize[1]
            W = 2*corner_radius + 2*border + tagSize[0]
            
            RGB = 255*np.ones((H,W,3))
            img = Image.fromarray(RGB.astype('uint8'))
            img = Image.new('RGBA',img.size,(255,255,255,0))
            draw = ImageDraw.Draw(img)
            
            # draw border rectangle
            draw.rectangle([(corner_radius+border,0),(corner_radius+tagSize[0]+border,H-1)],fill=_hex2RGB(border_color))
            draw.ellipse([(pad,pad-1),(2*corner_radius-pad+border,2*corner_radius-pad+border)],fill=_hex2RGB(border_color))
            draw.ellipse([(pad,H-2*corner_radius-pad-border),(2*corner_radius-pad+border,H-pad)],fill=_hex2RGB(border_color))
            draw.rectangle([(pad,pad+corner_radius+border),(pad+corner_radius+border,H-pad-corner_radius-border)],fill=_hex2RGB(border_color))
            draw.ellipse([(W-pad-2*corner_radius-border,pad-1),(W-pad,2*corner_radius+pad)],fill=_hex2RGB(border_color))
            draw.ellipse([(W-pad-2*corner_radius-border,H-2*corner_radius-pad),(W-pad,H-pad)],fill=_hex2RGB(border_color))
            draw.rectangle([(W-2*corner_radius-pad-border,pad+corner_radius),(W-pad,H-pad-corner_radius)],fill=_hex2RGB(border_color))
            
            
            # draw main rectangle
            draw.rectangle([(corner_radius + border,border),(corner_radius+tagSize[0]+border,H-border)],fill=_hex2RGB(tag_color))
            draw.ellipse([(pad+border,pad+border-1),(2*corner_radius-pad+border,2*corner_radius-pad+border)],fill=_hex2RGB(tag_color))
            draw.ellipse([(pad+border,H-2*corner_radius-pad-border),(2*corner_radius-pad+border,H-pad-border)],fill=_hex2RGB(tag_color))
            draw.rectangle([(pad+border,pad+corner_radius+border),(pad+corner_radius+border,H-pad-corner_radius-border)],fill=_hex2RGB(tag_color))
            draw.ellipse([(W-pad-2*corner_radius-border,pad+border-1),(W-pad-border,2*corner_radius+pad+border)],fill=_hex2RGB(tag_color))
            draw.ellipse([(W-pad-2*corner_radius-border,H-2*corner_radius-pad-border),(W-pad-border,H-pad-border)],fill=_hex2RGB(tag_color))
            draw.rectangle([(W-2*corner_radius-pad-border,pad+corner_radius+border),(W-pad-border,H-pad-corner_radius-border)],fill=_hex2RGB(tag_color))
            
            draw.text((corner_radius+border+pad, img.size[1]//2 - 0.7*tagSize[1]),txt,_hex2RGB(font_color),font=font)
            
            RGB = np.asarray(img)
            #plt.imshow(RGB)
            #plt.show()
            tagRGBs.append(RGB)
            tagSizes.append(RGB.shape[0:2])
                
        return tagSizes,tagRGBs
    
    def _set_arrows(RGB,line_radius=10,tagsPos=None,linesPos=None):
        
        if (linesPos is None) or (tagsPos is None):
            return RGB
        
        # dilation neighbourhood
        xv, yv = np.meshgrid(np.arange(-line_radius//2,line_radius//2 + 1),np.arange(-line_radius//2,line_radius//2 + 1) , sparse=False, indexing='ij')
        xv = xv.ravel()
        yv = yv.ravel()
        
        arrow_head = 4 # each side of the triangle will have a size of "arrow_head" times the border of the arrow line (line_radius)
        arrow_head *= line_radius
        
        for i,nodes in enumerate(linesPos):
            _from = tagsPos[nodes[0]]
            _to = tagsPos[nodes[1]]
            
            x0 = int(np.floor(np.mean(_from[[0,2]])))
            y0 = int(_from[3])
            
            x1 = int(np.floor(np.mean(_to[[0,2]])))
            y1 = int(_to[1])
            
            beta = 0.05
            tau = (y1+y0)//2
            yy = np.arange(y0,y1)
            xx = np.floor(np.abs(x1-x0)*sigmoid(yy,beta,tau) + np.min((x0,x1))).astype(int)
            if (x0 > x1):
                xx = xx[::-1]
            
            # Set the line of the arrow
            for k, (xi,yi) in enumerate(zip(xx,yy)):
                # local neighbourhood for dilation
                for ikj, (xoff,yoff) in enumerate(zip(xv,yv)):
                    RGB[yi+yoff,xi+xoff,0:-1] = 0
                    RGB[yi+yoff,xi+xoff,-1] = 255
            if (np.abs(x1-x0) < 10):
                # Now let's set the head of the arrow
                current_ncols = arrow_head
                arrow_y = int(np.floor(y1-np.sqrt(3)*arrow_head/2))
                while (current_ncols > 0):
                    RGB[arrow_y,x1-current_ncols//2:x1+current_ncols//2,:-1] = 0
                    RGB[arrow_y,x1-current_ncols//2:x1+current_ncols//2,-1] = 255
                    
                    current_ncols -= 1
                    #current_ncols = int(np.floor(current_ncols*ratio))
                    arrow_y += 1
        
        return RGB
    
    def _hex2RGB(h):
        if h[0] == "#":
            h = h[1:]
        R = int("0x"+h[0:2],16)
        G = int("0x"+h[2:4],16)
        B = int("0x"+h[4:6],16)
        A = 255
        if len(h) == 8:
            A = int("0x"+h[6:8],16)
        return R,G,B,A
    
    def sigmoid(x,beta=1,tau=0):
        return (1.)/(1.+np.exp(-(x-tau)*beta))


    # get the plain string with all nodes locations and sizes
    ddot = model_to_dot(model).create_plain().splitlines() # split linebreaks
    
    # Get the model layers
    layers = model.layers
    
    
    # Now, what we want, at the end, is a solid RGB image with all layers
    # represented as color tags. In order to do so we first need to create each
    # individual tag and render it. Each type of layer will be depicted using
    # a particular combination of fill color/ border color and text. 
    # The first step we need to do right now is to switch the type of each
    # layer so that we can define their tags (labels) and colors.
    
    # layerTypes: Variable to store the type of each layer (InputLayer, Conv2D, Concatenation, etc.)
    # layerLabels: Variable to store the label that we will print in the layer block, for each layer ("Input","Conv2D","Cat",etc.)
    # layerIds: Variable to store the index of each layer (so that we can identify them singularly) (0, 1, 2, etc.)
    layerTypes, layerLabels, layerIds = _get_layers_info(layers)
    
    
    # Now we are in a position to create each label (for each layer) 
    # individually.
    if (fontsize is None):
        fontsize = 210
    if (font is None):
        font = ImageFont.truetype("UbuntuMono-R.ttf", fontsize)
    else:
        if font is str:
            font = ImageFont.truetype(font,fontsize)
        elif type(font) is not PIL.ImageFont.FreeTypeFont:
            font = ImageFont.truetype("UbuntuMono-R.ttf", fontsize)
        
    top_margin = corner_radius//2
    tagSizes,tagRGBs = _print_layers(layers,layerTypes,font,fontsize,top_margin,border,corner_radius)
        
    # Get the vertical/horizontal scale so tags fit in the right position
    scale = _get_scale(ddot,layerIds,tagSizes)

    # Set tags positions
    tagsPos,linesPos,_W,_H = _set_tags_positions(ddot,layerIds)
      
    
    # Create Image and place each tag
    RGB = 255*np.ones((_H,_W,4)).astype('uint8')
    # set alpha layer to zero
    #RGB[:,:,-1] = 0
    for i,sz in enumerate(tagsPos):
        if (sz != []):
            RGB[sz[1]:sz[3],sz[0]:sz[2],:] = tagRGBs[i]
          
            
    
    # Now place the lines and arrows        
    line_radius = 15
    RGB = _set_arrows(RGB,line_radius,tagsPos,linesPos)
    
    if display:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,12))
        plt.imshow(RGB)
        plt.axis('off')
        plt.show()
    
    return RGB

   
