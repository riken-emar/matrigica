# -*- coding: utf-8 -*-
"""-------------------------------------------------------"""
" get material rigidity from NMR data (DQ, MAPE, Anisotropy) "
" Copyright from: 2021 Aug    "   
" CSRS, RIKEN Institute       "  
" Auther: A.KUROTANI, K.HARA  " 
"""-------------------------------------------------------"""

import nmrglue as ng
import scipy.signal as sig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy import integrate
from math import exp
import scipy
import matplotlib.cm as cm
import matplotlib as mpl
import sys
import os
import json
import supMaterialRigidity as supMR
from pulp import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

# global variable
x = np.zeros(1)
y = np.zeros(1)
saveFiles = ""
mdNo = 0 # 1:MAPE, 2:DQ, 3:Ani
dataT_D = np.zeros(1)
dataT_M = np.zeros(1)
dataT_A = np.zeros(1)
outputProcess = ""
isShowResSout = 1
saveNameFlag = "_p1"
isGUI = 1
isShowImage = 0 

##################
# commandline Str
##################
def expErr(memo):
    print (memo)
    print ("python", sys.argv[0], "[Dir-DQ] [Dir-MAPE] [Dir-Anisotropy] [dataSplitCnt(recommend 9 or 11)] [bayesIterCount(more than 5, unuse:-1)] [bayesCalcCount(example:3)] [savefile-Dir(str)] [savefile-Prefix(any str)] [freq.paramval-intermed(float)] [freq.paramval-soft(float)] [freq.paramval-hard(float)] [isCalcRateFor2D(0/1)] [isReCalc_inErr(0/1)] [isShowResult(0/1)]")
    print ("python", sys.argv[0], "./102 ./101 ./100 9 25 3 dataDir addCom 0.9 0.9 0.9 1 1 1")
    print ("python", sys.argv[0], "./202 ./201 ./200 11 20 2 dataDir addCom 0.9 0.9 0.9 0 1 1")
    exit()

########
# main
########
def main():
    args = sys.argv
    if len(args) < 14:
        expErr("Input argumentCnt is " + str(len(args)) + ".")
    else:
        global isShowResSout
        searchDir_TimeD = sys.argv[1]     # ReadDir-DQ
        searchDir_TimeM = sys.argv[2]     # ReadDir-MAPE
        searchDir_TimeA = sys.argv[3]     # ReadDir-Ani

        dataSplitCnt    = int(sys.argv[4]) # data splint Cnt.. - recommend 9 or 11
        bayes_iterCt    = int(sys.argv[5]) # IterCnt for bayesianOptimization (set 5-25, without:-1)
        bayes_repCnt    = int(sys.argv[6]) # execCnt bayesianOptimization (ex 3)
        saveDirName     = sys.argv[7]      # saveDir
        filePrefix      = sys.argv[8]      # prefix of save_fimename
        iniMed          = float(sys.argv[9])  # setting val. for freq fitting (intermediate): ex 0.2
        iniSoft         = float(sys.argv[10])  # setting val. for freq fitting (mobile-soft): ex 0.9
        iniHard         = float(sys.argv[11]) # setting val. for freq fitting (rigid-hard): ex 0.07
        isCalBayFor2D   = int(sys.argv[12])   # calc rate with bayes in 2D (0/1)
        isReCalc        = int(sys.argv[13])   # isRecalculate if fitting is error (0/1)
        isShowResSout   = int(sys.argv[14])   # isStdout of process & raito (0/1)

        bayes_iterCtEx = bayes_iterCt
        bayes_iterCt = bayes_iterCt-5         # Consistency of real no in bayesianOptimization

        if(os.path.isdir(searchDir_TimeD)==0):
            expErr("Input directry '" + searchDir_TimeD + "' is not exist.")
        elif(os.path.isdir(searchDir_TimeM)==0):
            expErr("Input directry '" + searchDir_TimeM + "' is not exist.")
        elif(os.path.isdir(searchDir_TimeM)==0):
            expErr("Input directry '" + searchDir_TimeA + "' is not exist.")
        
        #[Memo] saveNameFlag = "_pa1" # will be global var

    totalFlow(isCalBayFor2D, searchDir_TimeD, searchDir_TimeM, searchDir_TimeA, saveDirName, filePrefix, bayes_repCnt, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, iniSoft, iniMed, iniHard, saveNameFlag, isReCalc)

T2_mod1 = -1
T2_mod2 = -1
T2_ri1  = -1
T2_ri2  = -1
T2_ani1 = -1
T2_ani2 = -1
T2_ani3 = -1
T2_g = -1

########################################################
# fitting function --blackbox function (For time & freq)
########################################################
#---- for time ----#
def func_time(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        a = params[int(param_range[0])]
        W = params[int(param_range[1])]
        T2 = params[int(param_range[2])]
        y = y + a*np.exp(-(x/T2)**W)
        y_list.append(y)

    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i

    y_sum = y_sum + params[-1] # add background
    return y_sum

#---- for freq. ----#
def func_freq(x, *params): 
    num_func = int(len(params)/6)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(6*i,6*(i+1),1))
        amp1 = params[int(param_range[0])]
        ctr1 = params[int(param_range[1])]
        wid1 = params[int(param_range[2])]
        amp2 = params[int(param_range[3])]
        ctr2 = params[int(param_range[4])]
        wid2 = params[int(param_range[5])]
        if(amp1==0 or amp2==0):
            pass
        else:
            y += amp1*np.exp(-((x-ctr1)/wid1)**2) + amp2*wid2**2/((x-ctr2)**2+wid2**2) 
        y_list.append(y)

    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum += i
    
    y_sum += params[-1] # add background
    return y_sum


#######################################
# For plot of fitting (For time & freq)
#######################################
#---- for time ----#
def fit_plot_time(x, *params): # for time
    y_list = []

    num_func = int(len(params)/3)
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        a = params[int(param_range[0])]
        W = params[int(param_range[1])]
        T2 = params[int(param_range[2])]
        y = y + a*np.exp(-(x/T2)**W)  # 

        y_list.append(y)
    return y_list

#---- for freq. ----#
def fit_plot_freq(x, *params):
    y_list = []
    num_func = int(len(params)/6)
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(6*i,6*(i+1),1))
        amp1 = params[int(param_range[0])]
        ctr1 = params[int(param_range[1])]
        wid1 = params[int(param_range[2])]
        amp2 = params[int(param_range[3])]
        ctr2 = params[int(param_range[4])]
        wid2 = params[int(param_range[5])]
        y =(amp1*np.exp(-((x-ctr1)/wid1)**2) + amp2*wid2**2/((x-ctr2)**2+wid2**2)) # 

        y_list.append(y)
    return y_list


#################################################
# preperation of Time-fitting
#################################################
def preperFitTime(dicT, dataT, saveFileName, dataSplitCnt, iniSoft, iniMed, iniHard, saveNameFlag, bayes_iterCtEx, bayes_iterCt): 
    global saveFiles
    global x
    global y

    BF1=dicT['acqus']['BF1'] # 
    O1=dicT['acqus']['O1']   # 
    SW=dicT['acqus']['SW']

    y=dataT # 
    O1p=O1/BF1
    ppmmin=math.floor(int(O1p-(SW/2)))  # 
    ppmmax=math.floor(O1p+(SW/2))
    x=np.linspace(ppmmin,ppmmax,len(y)) # 
    X=''
    t=''

    # preperation 
    TD=dicT['acqus']['TD']
    at=TD/dicT['acqus']['SW_h']
    fs=TD/at

    calib=10 # 
    if TD>2048:
        nperseg=dataSplitCnt # TD/32
    else:
        dataSplitCnt = dataSplitCnt-2
        nperseg=TD/(dataSplitCnt) # 1024

    f,t, Zxxt = sig.stft(dataT[0:200], fs, nperseg=nperseg, return_onesided=False) # Short-time Fourier transform
    X = np.fft.fftshift((1+np.abs(Zxxt)),axes=0) # uppercase of X --Fourier transform with Numpy
    
    return X, t


#################################################
# freq fitting
#################################################
def fittingFreqRegionEle(optStren1, dicT, dataT, saveFileName, dataSplitCnt, iniSoft, iniMed, iniHard, saveNameFlag, bayes_iterCtEx, bayes_iterCt, fitMax_time): 
    global saveFiles
    global x
    global y

    BF1=dicT['acqus']['BF1'] # use this var in plt.pcolormesh()
    O1=dicT['acqus']['O1']   # use this var in plt.pcolormesh()
    SW=dicT['acqus']['SW']

    y=dataT # 
    O1p=O1/BF1
    ppmmin=math.floor(int(O1p-(SW/2)))  # 
    ppmmax=math.floor(O1p+(SW/2))
    x=np.linspace(ppmmin,ppmmax,len(y)) # lowercase x
    X=''
    t=''
    ckMemo = ""

    # List of ini
    guess = []
    # fitting in hard layer (Ani: hard-soft-med) # optStren1
    if(mdNo == 3): # in Ani.
        g_title = "Frequency (Anisotropy)"
        guess.append([iniHard, iniHard, iniHard, 0, 0, 0]) # 
        guess.append([iniMed, iniMed, iniMed, iniMed, iniMed, iniMed]) # 
        guess.append([0, 0, 0, iniSoft, iniSoft, iniSoft]) # ex. 0.9

    if(mdNo == 2): # DQ (hard - med.)
        g_title = "Frequency (DQ)"
        guess.append([iniHard, iniHard, iniHard, 0, 0, 0]) # ex. 0.2 
        guess.append([iniMed, iniMed, iniMed, iniMed, iniMed, iniMed]) # ex. 0.2

    elif(mdNo == 1): # MAPE (soft - med.)
        g_title = "Frequency (MAPE)"
        guess.append([iniMed, iniMed, iniMed, iniMed, iniMed, iniMed]) # ex. 0.2 
        guess.append([0, 0, 0, iniSoft, iniSoft, iniSoft]) # ex. 0.9

    background = 0 # background (dmy)
    guess_total = []
    for i in guess:
        guess_total.extend(i)
    guess_total.append(background)

    # curve fitting
    try:
        popt, pcov = curve_fit(func_freq, x, y, p0=guess_total) # 
    except RuntimeError as ex:
        popt, pcov = curve_fit(func_freq, x, y, p0=guess_total, maxfev=15000) 
    #popt, pcov = scipy.optimize.curve_fit(func_freq, x, y, p0=guess_total, maxfev=150000) # 
    fit = func_freq(x, *popt) # 
    totalMaxF = int(np.amax(fit))

    strenRate = fitMax_time/totalMaxF
    fit = fit*strenRate 
    y   = y*strenRate   # set intensity

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x, y, s=20, label="measured")
    plt.plot(x, fit , ls='-', c='black', lw=1, label="fitting")
    plt.xlabel('Frequency[kHz]', fontsize=10)
    plt.ylabel('Intensity', fontsize=10)
    plt.title(g_title)
    ax.set_xticks([-200, 0, 200])    # setting: x-axis
    y_list = fit_plot_freq(x, *popt) # 
    baseline = np.zeros_like(x) + popt[-1]
    for n,i in enumerate(y_list):
        i = i * strenRate  # set intensity
        if(mdNo != 3):     # except Ani.
            if(n==0):
                if(mdNo==1): # MAPE(soft)
                    labelStr = "mobile-intermediate" # intermediate
                    med_np = i.copy()
                else: # DQ(hard)
                    labelStr = "rigid"
                    rimo_np1 = i.copy()
            else:
                rimo_np0 = np.zeros(1) # dummy
                if(mdNo==1): # MAPE(soft)
                    labelStr = "mobile"
                    rimo_np1 = i.copy()
                else: # DQ(hard)
                    labelStr = "rigid-intermediate" # intermediate
                    med_np = i.copy()
        else: # Ani.
            if(n==0):
                rimo_np0 = i.copy() # rigid
                labelStr = "rigid"
            elif(n==1):
                med_np = i.copy() # intermediate
                labelStr = "intermediate"
            else:
                rimo_np1 = i.copy() # mobile 
                labelStr = "mobile"

        plt.fill_between(x, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6, label=labelStr) # rainbow color
        #cmap = supMR.gen_cmap_name(['r','fuchsia'])
        #plt.fill_between(x, i, baseline, facecolor=cmap(n/len(y_list)), alpha=0.6, label=labelStr)
    
    ckFit, ckMemo = supMR.checkGraphShape_freq(y, fit, med_np, rimo_np0, rimo_np1, x, mdNo, fitMax_time)

    if(isShowImage==1):
        if (ckFit==""):
            saveName = saveFileName+"_freqFit_OK"+saveNameFlag+".png"
        else:
            saveName = saveFileName+"_freqFit_NG"+ckFit+saveNameFlag+".png"
        ax.legend()
        fig.savefig(saveName)
        plt.close()
        saveFiles += "Saved: "+saveName + "\n"

    return X, t, totalMaxF, popt, y, med_np, rimo_np0, rimo_np1, fit, ckMemo, ckFit, iniSoft, iniMed, iniHard, strenRate


#########################
# Time domain fitting
#########################
def fittingTimeRegionEle(EleNo, X, t, saveFileName, mdNo, saveNameFlag, ri_npD_T, ri_med_npD_T, mo_npM_T, med_npM_T):
    global x
    global y
    
    y=X[EleNo,:]
    fig=plt.figure()
    plt.title("Elememnt "+str(EleNo)) # bar plot
    plt.plot(y)
    plt.close()
    x=t # 

    # Max val. get max-value from no1-2
    slop, intercept = supMR.makeLinearEquation(0, y[0], 1, y[1]) 
    # Max val. get max-value from no2-3
    #slop, intercept = supMR.makeLinearEquation(0, y[1], 1, y[2]) 
    totalMax = int(intercept)
    y1_tmp = y[0]
    y[0] = totalMax # 
    y_ori = y
    y_ori[0] = y1_tmp # setting y_ori

    # keep data #
    y_btm = supMR.getSpectrumT2(y)
    stB = y_btm[0]-4 # 
    enB = y_btm[0]-2 # 
    slop_med, intercept_med = supMR.makeLinearEquation(0, y[stB], 1, y[enB]) # 
    intercept_medi = int(intercept_med) 
    #optStren2 = intercept_medi
    #optT2_2 = y_btm[0]-1

    iniT2min = 0.000001 # 
    iniT2max = 0.000035 # 

    optStren1_realMax = totalMax
    optStren2 = totalMax/0.1

    optT2_1 = iniT2min
    optT2_2 = iniT2max
    optT2_3 = 0
    optT2_4 = 0

    optStren3 = 0
    optStren4 = 0
    if(ri_npD_T!=""):
        inten_mob  = np.amax(mo_npM_T)
        inten_mobi = np.amax(med_npM_T)
        inten_rig  = np.amax(ri_npD_T)
        inten_rigi = np.amax(ri_med_npD_T)
        t2_mob  = supMR.getSpectrumT2(mo_npM_T)
        t2_mobi = supMR.getSpectrumT2(med_npM_T)
        t2_rig  = supMR.getSpectrumT2(ri_npD_T)
        t2_rigi = supMR.getSpectrumT2(ri_med_npD_T)
        tot_TNo = mo_npM_T.shape[0]
        optStren1_realMax = inten_rig
        optStren2 = inten_mob
        optStren3 = inten_rigi
        optStren4 = inten_mobi
        optT2_1 = 0.001*t2_rig[0]/tot_TNo
        optT2_2 = 0.001*t2_mob[0]/tot_TNo
        optT2_3 = 0.001*t2_rigi[0]/tot_TNo
        optT2_4 = 0.001*t2_mobi[0]/tot_TNo

    guess_total, param_bounds = setParamBoundsTime(mdNo, optStren1_realMax, optStren2, optStren3, optStren4, optT2_1, optT2_2, optT2_3, optT2_4)
    try:
        popt, pcov = curve_fit(func_time, x, y, p0=guess_total, bounds=param_bounds) # 
    except ValueError as ex:
        optT2_1 = iniT2max/10
        optT2_2 = iniT2max
        guess_total, param_bounds = setParamBoundsTime(mdNo, optStren1_realMax, optStren2, optStren3, optStren4, optT2_1, optT2_2, optT2_3, optT2_4)
        try:
            popt, pcov = curve_fit(func_time, x, y, p0=guess_total, bounds=param_bounds) # 
        except ValueError as ex:
            optT2_1 = iniT2min
            optT2_2 = iniT2min*10
            guess_total, param_bounds = setParamBoundsTime(mdNo, optStren1_realMax, optStren2, optStren3, optStren4, optT2_1, optT2_2, optT2_3, optT2_4)
            popt, pcov = curve_fit(func_time, x, y, p0=guess_total, bounds=param_bounds) # 

    # Time domain Fitting & creating graph
    aveRMSE, fit_T, fitMax_time, ri_np_T, mo_np_T, med_np_T = timeFitCommon(x, y, y_ori, popt, pcov, saveFileName, EleNo, saveNameFlag)
    minTotRMSE = round(aveRMSE,2)
    return [EleNo, totalMax, optStren1_realMax, optStren2, optT2_1, optT2_2, minTotRMSE], popt, totalMax, optStren1_realMax, fitMax_time, optStren2, fit_T, ri_np_T, mo_np_T, med_np_T, y, y_ori


########################################################
# Setting of param_bounds for time domain fitting
########################################################
def setParamBoundsTime(mdNo, stren1, stren2, stren3, stren4, t2_1, t2_2, t2_3, t2_4):
    guess = []
    if(mdNo==3): # Ani
        guess.append([stren1, 2, t2_1])    # rigid [intensity, Weibull coefficient, T2]
        guess.append([stren2, 1, t2_2])    # mobile
        guess.append([stren3, 1.5, t2_3])  # intermed
    elif(mdNo==1): # MAPE
        guess.append([stren1, mdNo, t2_1*10]) #
        guess.append([stren2, 1.5,  t2_2*10]) # 
    else: # DQ
        guess.append([stren1, mdNo, t2_1]) # 
        guess.append([stren2, 1.5,  t2_2]) # 
    background = 5 # background
    guess_total = []
    for i in guess:
        guess_total.extend(i)
    guess_total.append(background)

    # fitting (hard + intermadiate)
    if(mdNo==3):# Ani
        #param_bounds=([stren1,1.999999,t2_1,stren2,1,t2_2,stren3,1.499999999,t2_3,stren4,1.499999999,t2_4,1],[np.inf,2,np.inf,np.inf,1.0000001,np.inf,np.inf,1.5,np.inf,np.inf,1.5,np.inf ,10])
        param_bounds=([stren1,1.999999,t2_1,stren2,1,t2_2,stren3,1.499999999,t2_3,1],[np.inf,2,np.inf,np.inf,1.0000001,np.inf,np.inf,1.5,np.inf ,10])
    elif(mdNo==1): # MAPE
        param_bounds=([stren1,mdNo-0.00001,0.00001,      stren1/3,1.4999999,t2_1,1],[np.inf,mdNo,np.inf,np.inf,1.5,np.inf,10]) # 
    else: # DQ
        param_bounds=([stren1,mdNo-0.00001,t2_1,      stren2,1.4999999,t2_2,1],[np.inf,mdNo,np.inf,np.inf,1.5,np.inf,10]) # 
    return guess_total, param_bounds


########################################################
# Time domain fitting <--common parts
# saveFileName(FILENAME) & EleNo(FILENO) are for graphs
# if saveFileName is empty, EleNo is dummy
########################################################
def timeFitCommon(x, y, y_ori, popt, pcov, saveFileName, EleNo, saveNameFlag):
    fit = func_time(x, *popt) # 
    rmseFit = np.sqrt(mean_squared_error(y, fit)) # 

    if(saveFileName!=""): # 
        fig=plt.figure()
        plt.title("Elememnt "+str(EleNo)+" (Accumulation)") # 
        plt.title("Elememnt "+str(EleNo)+" (Accumulation)") # 
        plt.scatter(x, y_ori, s=20, c='black', label="measured")
        plt.plot(x, fit, ls='-', c='black', lw=1, label="fitting")
        plt.xlabel('Time[s]', fontsize=10)
        plt.ylabel('Intensity', fontsize=10)

    
    y_list = fit_plot_time(x, *popt)
    baseline = np.zeros_like(x) + popt[-1]
    if(mdNo==1):   # MAPE
        g_title = "Time (MAPE)"
    elif(mdNo==2): # DQ
        g_title = "Time (DQ)"
    else:
        g_title = "Time (Anisotropy)"
    for n,i in enumerate(y_list):
        if(mdNo != 3): # except Ani
            if(n==0):
                #rimo_np1 = i.copy()
                if(mdNo==1): # MAPE(soft)
                    mo_np = i.copy()
                    ri_np = np.zeros(1) # dummy
                    labelStr = "mobile"
                else: # DQ(hard)
                    mo_np = np.zeros(1) # dummy
                    ri_np = i.copy()
                    labelStr = "rigid"
            else:
                med_np = i.copy()
                if(mdNo==1): # MAPE(soft)
                    labelStr = "mobile-intermediate" # intermed.
                else:
                    labelStr = "rigid-intermediate" # intermed.
        else: # Ani
            if(n==0):
                mo_np = i.copy() # mobile (soft)
                labelStr = "mobile"
            elif(n==1):
                ri_np = i.copy() # rigid (hard)
                labelStr = "rigid"
            else: # 
                med_np = i.copy() # intermed.
                labelStr = "mobile-intermediate"
        
        plt.fill_between(x, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6, label=labelStr) # rainbow color
        #cmap = supMR.gen_cmap_name(['r','fuchsia'])
        #plt.fill_between(x, i, baseline, facecolor=cmap(n/len(y_list)), alpha=0.6, label=labelStr)

    if(saveFileName!="" and isShowImage==1):
        plt.title(g_title)
        plt.legend()
        saveName = saveFileName+"_timeFit_Ele"+ str(EleNo)+saveNameFlag+".png"
        fig.savefig(saveName)
        plt.close()
        global saveFiles
        saveFiles += "Saved: "+saveName + "\n"

    maxEles1 = np.maximum(ri_np, mo_np) 
    maxEles = np.maximum(maxEles1,med_np)
    rmseEle = np.sqrt(mean_squared_error(y, maxEles)) #  RMSE between y & Max
    aveRMSE = (rmseFit + rmseEle)/2
    fitMax_time = np.amax(fit)

    return aveRMSE, fit, fitMax_time, ri_np, mo_np, med_np


##############################################
# Display summary (proccess)
##############################################
def keepProcessResult(isRun, cate, searchDir_Freq, searchDir_Time, ckMemo, bayes_iterCtEx, dataSplitCnt, resTimes, catCntStr, iniSoft, iniMed, iniHard, totalMax):
    if(catCntStr==""):
        ex_catCntStr = ""
    else:
        ex_catCntStr = catCntStr

    expStr = "************ ["+cate+" "+ex_catCntStr+"] ************\n"
    expStr += "-- Search Dir --\n"
    expStr += "[Freq]: "+ str(searchDir_Freq) +"\n"
    expStr += "[Time]: "+ str(searchDir_Time) +"\n"

    if (isRun):
        expStr += "\n-- Bayesian optimization for Fitting (Freq-region, Time-region, Anisotropy) --\n"
        if(bayes_iterCtEx==-1):
            expStr += "un-use\n"
        else:        
            expStr += "Done\n Iteration Counts: " + str(bayes_iterCtEx)+"\n"

        expStr += "\n-- Parameter values of frequency-region fitting --\n"
        expStr += "MaxStrength:"+ str(totalMax) + "\n"

        expTit = ["Soft", "InterMediate", "Hard"]
        expVal = [iniSoft, iniMed, iniHard]
        expStr += '\t'.join(expTit)+"\n"
        expStr += '\t'.join(map(str, expVal))+"\n"

        expStr += "\n-- Graph shape of frequency on resion-fitting --\n"
        expStr += ckMemo + "\n"
        
        expStr += "\n-- Results of time-region fitting --\n"
        expTit = ["EleNo(/"+str(dataSplitCnt)+")", "FreqPeakMax", "Stren1(MAPE,DQ)", "Stren2(interMediate)", "T2(MAPE)", "T2(DQ)", "minAveRMSE"]
        
        expStr += '\t'.join(expTit)+"\n"
        for vars in resTimes:
            expStr += '\t'.join(map(str, vars))+"\n"

        global saveFiles
        expStr += "\n-- Saved graph files --\n"
        expStr += saveFiles+"\n"
        saveFiles = ""

    global outputProcess
    outputProcess += expStr
    if(isShowResSout):
        print (expStr)

##############################################
# Display summary (Ratio)
##############################################
def showResultSummay(treedRData, freqRD_List, timeRD_List, bayes_iterCtEx, saveDirName, filePrefix, searchDir_TimeA, searchDir_TimeD, ckDataStr, dominanceStr, ckDataGraph):
    trList =[str(trele) for trele in treedRData]
    if (bayes_iterCtEx < 5):
        bayes_iterCtEx = 5

    if (ckDataGraph==""):
        ckDataGraph = "OK"
    else:
        ckDataGraph = ckDataGraph.replace("\n",",")
        ckDataGraph = ckDataGraph.strip()

    expStr = "-- Ratio of mobile/rigid --\n"
    expStr += "mobile"+"\t"
    expStr += "rigid"+"\t"
    if(isGUI==1): # isGUI: global
        expStr += "\n"+trList[0]+"\t"+trList[1]
        expStr += "\n\nmobile"+"\t"
        expStr += "mobile-intermediate"+"\t"
        expStr += "rigid"+"\t"
        expStr += "rigid-intermediate"+"\n"
        expStr += trList[2]+"\t"+trList[3]+"\t"+trList[4]+"\t"+trList[5]+"\n"
        expStr += "\n[memo]"+"\n"
        expStr += ckDataStr
    elif(isGUI==-1):
        fList =[str(frele) for frele in freqRD_List]
        tiList =[str(tiele) for tiele in timeRD_List]
        frData = "\t".join(fList)
        tiData = "\t".join(tiList)
        treData = "\t".join(trList)
        expStr = "[(Freq)mobile:rigid]\t\t"
        expStr += "[mobile:mobile-intermed:rigid:rigid-intermed]\t\t\t\t"
        expStr += "[Ani-mobile:intermed:rigid]\t\t\t"
        expStr += "[(Time)mobile:rigid]\t\t"
        expStr += "[mobile:mobile-intermediate:rigid:rigid-intermed]\t\t\t\t"
        expStr += "[Ani-mobile:intermed:rigid]\t\t\t"
        expStr += "[(3D)mobile:rigid]\t\t"
        expStr += "[mobile:mobile-intermed:rigid:rigid-intermed]\t\t\t\t"
        expStr += "[Ani-mobile:intermed:rigid]\n"
        expStr += "mobile"+"\t"
        expStr += "rigid"+"\t"
        expStr += "mobile"+"\t"
        expStr += "mobile-inter"+"\t"
        expStr += "rigid"+"\t"
        expStr += "rigid-inter"+"\t"
        expStr += "mobile"+"\t"
        expStr += "inter"+"\t"
        expStr += "rigid"+"\t"
        expStr += "mobile"+"\t"
        expStr += "rigid"+"\t"
        expStr += "mobile"+"\t"
        expStr += "mobile-inter"+"\t"
        expStr += "rigid"+"\t"
        expStr += "rigid-inter"+"\t" 
        expStr += "mobile"+"\t"
        expStr += "inter"+"\t"
        expStr += "rigid"+"\t"
        expStr += "mobile"+"\t"
        expStr += "rigid"+"\t"
        expStr += "mobile"+"\t"
        expStr += "mobile-inter"+"\t"
        expStr += "rigid"+"\t"
        expStr += "rigid-inter"+"\t"
        expStr += "mobile"+"\t"
        expStr += "inter"+"\t"
        expStr += "rigid"+"\t"
        expStr += "check data, graph etc."+"\t"
        expStr += "bayes_iterCounts"+"\t"
        expStr += "bayes calc times(dominance/total)"+"\t"
        expStr += "memo"+"\n"
        expStr += frData + "\t"
        expStr += tiData + "\t"
        expStr += treData + "\t"
        expStr += ckDataGraph+"\t"       
        expStr += str(bayes_iterCtEx)+"\t" #
        expStr += dominanceStr+"\t" #
        expStr += ckDataStr + "\n"

        '''
        fList =[str(frele) for frele in freqRData]
        tiList =[str(tiele) for tiele in timeRData]
        trList =[str(trele) for trele in treedRData]
        if (bayes_iterCtEx < 0):
            bayes_iterCtEx = 0
        #frData = "\t".join(fList)
        #tiData = "\t".join(tiList)
        treData = "\t".join(trList)
        #expStr = "Frequency\t\t\t\t\t\t"
        #expStr += "Time\t\t\t\t\t\t"
        #expStr += "Frequency&Time\n"
        expStr = "[mobile:rigid]\t\t"
        expStr += "[mobile:mobile-intermediate:rigid:rigid-intermediate]\t\t\t\t\n"
        
        expStr += "[mobile:rigid]\t\t"
        expStr += "[mobile:mobile-intermediate:rigid:rigid-intermediate]\t\t\t\t"
        expStr += "[mobile:rigid]\t\t"
        expStr += "[mobile:mobile-intermediate:rigid:rigid-intermediate]\n"
        expStr += "mobile"+"\t"
        expStr += "rigid"+"\t"
        expStr += "mobile"+"\t"
        expStr += "mobile-intermediate"+"\t"
        expStr += "rigid"+"\t"
        expStr += "rigid-intermediate"+"\t"
        expStr += "mobile"+"\t"
        expStr += "rigid"+"\t"
        expStr += "mobile"+"\t"
        expStr += "mobile-intermediate"+"\t"
        expStr += "rigid"+"\t"
        expStr += "rigid-intermediate"+"\t" 
        
        expStr += "mobile"+"\t"
        expStr += "rigid"+"\t"
        expStr += "mobile"+"\t"
        expStr += "mobile-intermediate"+"\t"
        expStr += "rigid"+"\t"
        expStr += "rigid-intermediate"+"\t"

        expStr += "check data, graph etc."+"\t"
        expStr += "bayes-iterCounts"+"\t"
        expStr += "bayes calc times(dominance/total)"+"\t"
        expStr += "memo"+"\n"

        expStr += frData + "\t"
        expStr += tiData + "\t"
        expStr += treData + "\t"
        expStr += ckDataGraph+"\t"
        expStr += str(bayes_iterCtEx)+"\t" #
        expStr += dominanceStr+"\t" #
        expStr += ckDataStr + "\n"
        '''
    else:
        treData = "\t".join(trList)
        expStr += "mobile"+"\t"
        expStr += "mobile-intermediate"+"\t"
        expStr += "rigid"+"\t"
        expStr += "rigid-intermediate"+"\t"
        expStr += "check data, graph etc."+"\t"
        expStr += "bayes-iterCounts"+"\t"
        expStr += "bayes calc times(dominance/total)"+"\t"
        expStr += "memo"+"\n"
        expStr += treData + "\t"
        expStr += ckDataGraph+"\t"
        expStr += str(bayes_iterCtEx)+"\t" #
        expStr += dominanceStr+"\t" #
        expStr += ckDataStr + "\n"

    saveName = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeA) + "-" + os.path.basename(searchDir_TimeD) + "_summary_Ratio.txt"
    f = open(saveName, 'w')
    f.write(expStr)


##############################################
# Display summary (Ratio)
##############################################
def showResultRatio(dominanceStr, ckDataStr, dmrateA_D, dmrateA_M, ckData, isCompRate, cate, mob_mape, inter_mape, rigid_dq, inter_dq, aniTotal, mob_mapeRate, inter_mapeRate, rigid_dqRate, inter_dqRate, aniTotalRate, rateXmape, rateYdq, op_mapeRate2, op_dqRate2, b2_mob_mape, b2_mob_mape_med, b2_rig_dq, b2_rig_dq_med, isDQexist, isMAPEexist, isAniexist, isShapeD_freq, isShapeM_freq, isRunD, isRunM_freq, mob_ARate10, med_ARate10, rig_ARate10, dq_rig_Rate, dq_rigMed_Rate, ma_mob_Rate, ma_mobMed_Rate):
    expStr = "************ ["+cate+"] ************\n"

    expStr += "--------- Data existance, Calculation, Graph shape ---------\n"
    if(ckData==""):
        expStr += "OK\n" 
    else:
        expStr += ckData+"\n" 

    if (dmrateA_D!=""):
        expStr += "\n -- Adjustment intensity scale of MAPE to Anisotropy, & DQ to Anisotropy --\n"
        expStr += "MAPE to Anisotropy: "
        expStr += str(round(dmrateA_M,3)) + " times\n"
        expStr += "DQ to Anisotropy: "
        expStr += str(round(dmrateA_D,3)) + " times\n"

    expStr += "\n --------- Component ratio with the volume with chart graph ---------\n"
    expStr += "[mobile]:[mobile-intermediate]:[rigid]:[rigid-intermediate]:[aniTotalRate]\n"
    if(isCompRate==1):
        expStr += str(mob_mape)+" "+str(inter_mape)+" "+str(rigid_dq)+" "+str(inter_dq)+" "+str(aniTotal)+"\n"
        expStr += str(mob_mapeRate)+" "+str(inter_mapeRate)+" "+str(rigid_dqRate)+" "+str(inter_dqRate)+" "+str(aniTotalRate)+"\n"

        expStr += "\n -- Component ratio with pulp method --\n"
        expStr += "[mobile, mobile-intermediate]:[rigid, rigid-intermediate]\n"
        expStr += str(rateXmape)+"\t"+str(rateYdq)+"\n"

        expStr += "\n -- Component ratio with bayesoptimization --\n"
        expStr += "[mobile]:[rigid]\n"
        expStr += str(op_mapeRate2)+"\t"+str(op_dqRate2)+"\n"
        expStr += "[mobile, mobile-intermediate]:[rigid, rigid-intermediate]\n"
        #expStr += str(b2_mob_mape)+"%\t"+str(b2_mob_mape_med)+"%\t"+str(b2_rig_dq)+"%\t"+str(b2_rig_dq_med)+"%\n"
        expStr += str(ma_mob_Rate)+"\t"+str(ma_mobMed_Rate)+"\t"+str(dq_rig_Rate)+"\t"+str(dq_rigMed_Rate)+"\t"+ckDataStr+"\n"
        expStr += "dominance calc. times: " + dominanceStr+"\n"

        expStr += "\n -- Component ratio from Anisotropy data --\n"
        expStr += "[mobile : intermediate : rigid]\n"        
        expStr += str(mob_ARate10)+"\t"+str(med_ARate10)+"\t"+str(rig_ARate10)+"\n"

        global saveFiles
        expStr += "\n -- Saved graph files --\n"
        expStr += saveFiles+"\n"
        saveFiles = "" 
    else:
        if(isDQexist==0):
            expStr += "DQ data doesn't exist.\n"
        if(isMAPEexist==0):
            expStr += "MAPE data doesn't exist.\n"
        if(isAniexist==0):
            expStr += "Anisotropy data doesn't exist.\n"
        if(isShapeD_freq==0):
            expStr += "The graph shape from DQ is not good.\n"
        if(isShapeM_freq==0):
            expStr += "The graph shape from MAPE is not good.\n"
        if(isRunD==0):
            expStr += "The calculation with DQ is failed.\n"
        if(isRunM_freq==0):
            expStr += "The calculation with MAPE is failed.\n"
        expStr += "*** The component ratio is no assignment.\n"
    if(isShowResSout):
        print (expStr)
    return expStr


###########################################################
# Fitting -Time & Freq
# ---------------------------------------------------------
# iniSoft: soft layer val
# iniMed:  intermed layer val
# iniHard: hard layer val
# cate:    : category let.
# dicT: dic of Time
# dataT: npdata of Time
# dicF: dic of Freq
# dataF: npdata of Freq
# bayes_iterCt: bayes's iter Cnt.  (as command counts)
# bayes_iterCtEx: bayes's iter Cnt.  (as real counts)
# dataSplitCnt: splint counts in time fitting data
# mdNo: MAPE:1, DQ:2, Ani:3
# searchDir_Freq: directory name for freq data
# searchDir_Time: directory name for time data
# saveNameFlag: information string for saveFileName
# catCnt: category no
# ri_npD_T, ri_med_npD_T, mo_npM_T, med_npM_T: each npdata
###########################################################
def execFitting_Time_and_Freq(iniSoft, iniMed, iniHard, cate, dicT, dataT, dicF, dataF, saveFileName, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_Freq, searchDir_Time, saveNameFlag, catCnt, ri_npD_T, ri_med_npD_T, mo_npM_T, med_npM_T):

    ## Fitting (time) ##
    #-- preparation --：X(after Fourier transform),t(after short time Fourier transform)
    X, t = preperFitTime(dicT, dataT, saveFileName, dataSplitCnt, 0,0,0, saveNameFlag, bayes_iterCtEx, bayes_iterCt)
    resTimes = []
    fitNo = math.floor(dataSplitCnt/2) # 
    #-- Fitting (time) --:with only a middle data of split data
    resTmed, popt_T, totalMaxT, optStren1_realMax, fitMax_time, optStren2, fit_T, ri_np_T, mo_np_T, med_np_T, y_T, y_ori_T = fittingTimeRegionEle(fitNo, X, t, saveFileName, mdNo, saveNameFlag, ri_npD_T, ri_med_npD_T, mo_npM_T, med_npM_T)
    resTimes.append(resTmed)
   
    ## Fitting (freq) ##
    isRun_freq = 1
    isShape_freq = 0

    try:
        X_dmy1, t_dmy1, totalMaxF, poptFreq, y_freq, med_np_freq, rimo_np0_freq, rimo_np1_freq, fit_freq, ckMemo, ckFit, iniSoft, iniMed, iniHard, strenRate = fittingFreqRegionEle(optStren1_realMax, dicF, dataF, saveFileName, dataSplitCnt, iniSoft, iniMed, iniHard, saveNameFlag, bayes_iterCtEx, bayes_iterCt, fitMax_time)
        if(ckFit==""):
            isShape_freq=1

    except RuntimeError as ex:
        cate = cate + " " + str(type(ex)) + " " + str(ex)
        isRun_freq = 0
        totalMaxF = 0
        strenRate = 0
        poptFreq = np.zeros(1)      # poptDmy
        med_np_freq =  np.zeros(1)  # med_npDmy
        rimo_np0_freq = np.zeros(1) # rimo_np0Dmy
        rimo_np1_freq = np.zeros(1) # rimo_np1Dmy
        fit_freq = np.zeros(1) # fitDmy
        y_freq = np.zeros(1)   # yDmy
        ckMemo = "" # memoDmy
        ckFit = ""  #ckFitDmy

    # keep process data
    keepProcessResult(isRun_freq, cate, searchDir_Freq, searchDir_Time, ckMemo, bayes_iterCtEx, dataSplitCnt, resTimes, str(catCnt), iniSoft, iniMed, iniHard, totalMaxF)
    return poptFreq, popt_T, y_freq, med_np_freq, rimo_np0_freq, rimo_np1_freq, fit_freq, isShape_freq, isRun_freq, fit_T, ri_np_T, mo_np_T, med_np_T, y_T, y_ori_T, strenRate


##############################################
# total flow of fitting 
# --------------------------------------------
# searchDir_TimeD: Dir of DQ data
# searchDir_TimeM: Dir of MAPE data
# searchDir_TimeA: Dir of Ani data
# saveDirName: savedir
# bayes_iterCt: bayes's iter Cnt.  (as command counts)
# bayes_iterCtEx: bayes's iter Cnt.  (as real counts)
# dataSplitCnt: splint counts in time fitting data
# iniSoft, iniMed, iniHard: parameter values (in freq domain)
# saveNameFlag: supporting word for savefilename
# isReCalc(0/1): recalc after err
##############################################
def totalFlow(isCalBayFor2D, searchDir_TimeD, searchDir_TimeM, searchDir_TimeA, saveDirName, filePrefix, bayes_repCnt, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, iniSoft, iniMed, iniHard, saveNameFlag, isReCalc):
    global mdNo
    global dataT_D
    global dataT_M
    global dataT_A
    global dataF_D
    global dataF_M
    global dataF_A

    #---- [DQ] Read Freq & Time ----#
    cate = "DQ"
    mdNo = 2
    isDQexist = 1
    isShapeD_freq  = 0
    isRunD_freq = 0
    ri_npD_freq = np.zeros(1)   
    med_np_D_freq = np.zeros(1)
    popt_DQ_Freq = np.zeros(1)
    poptDQ_T  = np.zeros(1)
    fitD_T = np.zeros(1)
    fitM_T = np.zeros(1)
    inSMH_0 = [iniSoft, iniMed, iniHard]

    searchDir_FreqD = searchDir_TimeD + '/pdata/1'
    try:
        dicT_D, dataT_D = ng.fileio.bruker.read(dir=searchDir_TimeD)       # Read DQ time domain (-Rigid)
        dataT_D = ng.bruker.remove_digital_filter(dicT_D, dataT_D)         # Filteration
    except IOError as ex:
        cate = cate + " " + str(type(ex)) + " " + str(ex)
        keepProcessResult(0, cate, searchDir_FreqD, "", "", 0, 0, [], "", 0, 0, 0, 0) # keep process data
        isDQexist = 0
    if(isDQexist):
        try:
            dicF_D, dataF_D = ng.fileio.bruker.read_pdata(dir=searchDir_FreqD) # Read DQ freq domain (-Rigid)
            saveBaseN_D = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeD) + cate

            #--- Time & Freq Fitting ---#
            popt_DQ_Freq, poptDQ_T, y_D_freq, med_np_D_freq, rimoDmy, ri_npD_freq, fit_D_freq, isShapeD_freq, isRunD_freq, fitD_T, ri_npD_T, mo_npD_TDmy, ri_med_npD_T, y_npD_T, yori_npD_T, strenRateD = execFitting_Time_and_Freq(inSMH_0[0], inSMH_0[1], inSMH_0[2], cate, dicT_D, dataT_D, dicF_D, dataF_D, saveBaseN_D, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqD, searchDir_TimeD, saveNameFlag, 1,"","","","")
            if(isReCalc==1 and (isShapeD_freq==0 or isRunD_freq==0)): # check graph shape is NG or iterN. Recalculation with 1/9 value of params, if err.
                saveNameFlagD = saveNameFlag.rstrip("1")+"2" # 
                inSMH = list(map(lambda x: x/9, inSMH_0))
                popt_DQ_Freq, poptDQ_T, y_D_freq, med_np_D_freq, rimoDmy, ri_npD_freq, fit_D_freq, isShapeD_freq, isRunD_freq, fitD_T, ri_npD_T, mo_npD_TDmy, ri_med_npD_T, y_npD_T, yori_npD_T, strenRateD = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_D, dataT_D, dicF_D, dataF_D, saveBaseN_D, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqD, searchDir_TimeD, saveNameFlagD, 2,"","","","")
                if(isReCalc==1 and (isShapeD_freq==0 or isRunD_freq==0)): # check graph shape is NG or iterN. Recalculation with 1/9 value of params, if err.
                    saveNameFlagD = saveNameFlagD.rstrip("2")+"3" # 
                    inSMH = list(map(lambda x: x/10, inSMH_0))
                    popt_DQ_Freq, poptDQ_T, y_D_freq, med_np_D_freq, rimoDmy, ri_npD_freq, fit_D_freq, isShapeD_freq, isRunD_freq, fitD_T, ri_npD_T, mo_npD_TDmy, ri_med_npD_T, y_npD_T, yori_npD_T, strenRateD = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_D, dataT_D, dicF_D, dataF_D, saveBaseN_D, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqD, searchDir_TimeD, saveNameFlagD, 3,"","","","")
                    if(bayes_iterCtEx>=5 and (isShapeD_freq==0 or isRunD_freq==0)): # recalculation, if NG data. 
                        saveNameFlagD = saveNameFlagD.rstrip("3")+"4" # 
                        inSMH = list(map(lambda x: x/11, inSMH_0))
                        popt_DQ_Freq, poptDQ_T, y_D_freq, med_np_D_freq, rimoDmy, ri_npD_freq, fit_D_freq, isShapeD_freq, isRunD_freq, fitD_T, ri_npD_T, mo_npD_TDmy, ri_med_npD_T, y_npD_T, yori_npD_T, strenRateD = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_D, dataT_D, dicF_D, dataF_D, saveBaseN_D, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqD, searchDir_TimeD, saveNameFlagD, 3,"","","","")
                        if(bayes_iterCtEx>=5 and (isShapeD_freq==0 or isRunD_freq==0)): # recalculation, if NG data. 
                            saveNameFlagD = saveNameFlagD.rstrip("4")+"5" # 
                            inSMH = list(map(lambda x: x*1.055, inSMH_0))
                            popt_DQ_Freq, poptDQ_T, y_D_freq, med_np_D_freq, rimoDmy, ri_npD_freq, fit_D_freq, isShapeD_freq, isRunD_freq, fitD_T, ri_npD_T, mo_npD_TDmy, ri_med_npD_T, y_npD_T, yori_npD_T, strenRateD = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_D, dataT_D, dicF_D, dataF_D, saveBaseN_D, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqD, searchDir_TimeD, saveNameFlagD, 4,"","","","")

        except IOError as ex:
            cate = cate + " " + str(type(ex)) + " " + str(ex)
            keepProcessResult(0, cate, searchDir_FreqD, "", "", 0, 0, [], "", 0, 0, 0, 0) # keep process data
            isDQexist = 0    

    #---- [MAPE] Read Freq & Time fitting ----#
    mdNo = 1
    cate = "MAPE"
    isMAPEexist = 1
    isShapeM_freq = 0
    isRunM_freq = 0
    searchDir_FreqM = searchDir_TimeM + '/pdata/1'
    try:
        dicT_M, dataT_M = ng.fileio.bruker.read(dir=searchDir_TimeM) # Read MAPE domain (-Mobile)
        dataT_M = ng.bruker.remove_digital_filter(dicT_M, dataT_M)   # Filteration
    except IOError as ex:
        cate = cate + " " + str(type(ex)) + " " + str(ex)
        keepProcessResult(0, cate, searchDir_FreqM, "", "", "", 0, [], "", 0, 0, 0, 0) # keep process data
        isMAPEexist = 0

    if(isMAPEexist):
        try:
            dicF_M, dataF_M = ng.fileio.bruker.read_pdata(dir=searchDir_FreqM) # Read MAPE domain (-Mobile)
            #dataF_M = ng.bruker.remove_digital_filter(dicF_M, dataF_M)         # Filteration

            #--- Time & Freq Fitting ---#
            saveBaseN_M = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeM) + cate
            popt_MAPE_Freq, poptMAPE_T, y_M, med_np_M_freq, rimoDmy, mo_npM_freq, fit_M_freq, isShapeM_freq, isRunM_freq, fitM_T, ri_npM_TDmy, mo_npM_T, med_npM_T, y_npM_T, yori_npM_T, strenRateM = execFitting_Time_and_Freq(inSMH_0[0], inSMH_0[1], inSMH_0[2], cate, dicT_M, dataT_M, dicF_M, dataF_M, saveBaseN_M, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqM, searchDir_TimeM, saveNameFlag, 1,"","","","")
            if(isReCalc==1 and (isShapeM_freq==0 or isRunM_freq==0)): # [check] graph shape is NG or iterN. Recalculation with 1/9 value of params, if err.
                saveNameFlagM = saveNameFlag.rstrip("1")+"2" # 
                inSMH = list(map(lambda x: x*9, inSMH_0)) 
                popt_MAPE_Freq, poptMAPE_T, y_M, med_np_M_freq, rimoDmy, mo_npM_freq, fit_M_freq, isShapeM_freq, isRunM_freq, fitM_T, ri_npM_TDmy, mo_npM_T, med_npM_T, y_npM_T, yori_npM_T, strenRateM = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_M, dataT_M, dicF_M, dataF_M, saveBaseN_M, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqM, searchDir_TimeM, saveNameFlagM, 2,"","","","")
                if(isReCalc==1 and (isShapeM_freq==0 or isRunM_freq==0)): # [check] graph shape is NG or iterN. Recalculation with 1/9 value of params, if err.
                    saveNameFlagM = saveNameFlagM.rstrip("2")+"3" # 
                    inSMH = list(map(lambda x: x*10, inSMH_0)) 
                    popt_MAPE_Freq, poptMAPE_T, y_M, med_np_M_freq, rimoDmy, mo_npM_freq, fit_M_freq, isShapeM_freq, isRunM_freq, fitM_T, ri_npM_TDmy, mo_npM_T, med_npM_T, y_npM_T, yori_npM_T, strenRateM = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_M, dataT_M, dicF_M, dataF_M, saveBaseN_M, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqM, searchDir_TimeM, saveNameFlagM, 3,"","","","")
                    if(bayes_iterCtEx>=5 and (isShapeM_freq==0 or isRunM_freq==0)): # recalculation, if NG data.
                        saveNameFlagM = saveNameFlagM.rstrip("3")+"4" # 
                        inSMH = list(map(lambda x: x*11, inSMH_0)) 
                        popt_MAPE_Freq, poptMAPE_T, y_M, med_np_M_freq, rimoDmy, mo_npM_freq, fit_M_freq, isShapeM_freq, isRunM_freq, fitM_T, ri_npM_TDmy, mo_npM_T, med_npM_T, y_npM_T, yori_npM_T, strenRateM = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_M, dataT_M, dicF_M, dataF_M, saveBaseN_M, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqM, searchDir_TimeM, saveNameFlagM, 3,"","","","")
                        if(bayes_iterCtEx>=5 and (isShapeM_freq==0 or isRunM_freq==0)): # recalculation, if NG data.
                            saveNameFlagM = saveNameFlagM.rstrip("4")+"5" #
                            inSMH = list(map(lambda x: x*1.055, inSMH_0)) 
                            popt_MAPE_Freq, poptMAPE_T, y_M, med_np_M_freq, rimoDmy, mo_npM_freq, fit_M_freq, isShapeM_freq, isRunM_freq, fitM_T, ri_npM_TDmy, mo_npM_T, med_npM_T, y_npM_T, yori_npM_T, strenRateM = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_M, dataT_M, dicF_M, dataF_M, saveBaseN_M, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqM, searchDir_TimeM, saveNameFlagM, 4,"","","","")
        except IOError as ex:
            cate = cate + " " + str(type(ex)) + " " + str(ex)
            keepProcessResult(0, cate, searchDir_FreqM, "", "", 0, 0, [], "", 0, 0, 0, 0) # keep process data
            isMAPEexist = 0
    #print(1005)
    #exit()

    #---- [Ani] Read Freq & Time fitting  ----#
    mdNo = 3
    cate = "Ani"
    isAniexist = 1
    isRunA_freq = 0
    dataF_A = np.zeros(1) 
    #y_npA_T = np.zeros(1) 
    searchDir_FreqA = searchDir_TimeA + '/pdata/1'

    try:
        dicT_A, dataT_A = ng.fileio.bruker.read(dir=searchDir_TimeA)       # Read time domain data
        dataT_A = ng.bruker.remove_digital_filter(dicT_A, dataT_A)         # Filteration
    except IOError as ex:
        cate = cate + " " + str(type(ex)) + " " + str(ex)
        keepProcessResult(0, cate, searchDir_FreqA, "", "", 0, 0, [], "", 0, 0, 0, 0) # keep process data
        isAniexist = 0

    if(isAniexist):
        try:
            dicF_A, dataF_A = ng.fileio.bruker.read_pdata(dir=searchDir_FreqA) # Read freq domain data
            #dataF_A = ng.bruker.remove_digital_filter(dicF_A, dataF_A)         # Filteration

            #--- Time & Freq Fitting ---#
            saveBaseN_I = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeA) + cate # 
            popt_Ani_Freq, poptAni_T, y_A_F, med_npA_F, mo_npA_F, ri_npA_F, fitA_F, isShapeA_freq, isRunA_freq, fitA_T, ri_npA_T, mo_npA_T, med_npA_T, y_npA_T, yori_npA_T, strenRateA = execFitting_Time_and_Freq(inSMH_0[0], inSMH_0[1], inSMH_0[2], cate, dicT_A, dataT_A, dicF_A, dataF_A, saveBaseN_I, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqA, searchDir_TimeA, saveNameFlag, 1, ri_npD_T, ri_med_npD_T, mo_npM_T, med_npM_T)
            if(isReCalc==1 and (isShapeA_freq==0 or isRunA_freq==0)): # [check] graph shape is NG or iterN. Recalculation with 1/9 value of params, if err.
                saveNameFlagA = saveNameFlag.rstrip("1")+"2" # 
                inSMH = list(map(lambda x: x*9, inSMH_0)) 
                popt_Ani_Freq, poptAni_T, y_A_F, med_npA_F, mo_npA_F, ri_npA_F, fitA_F, isShapeA_freq, isRunA_freq, fitA_T, ri_npA_T, mo_npA_T, med_npA_T, y_npA_T, yori_npA_T, strenRateA = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_A, dataT_A, dicF_A, dataF_A, saveBaseN_I, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqA, searchDir_TimeA, saveNameFlagA, 2, ri_npD_T, ri_med_npD_T, mo_npM_T, med_npM_T)
                if(isReCalc==1 and (isShapeA_freq==0 or isRunA_freq==0)): # [check] graph shape is NG or iterN. Recalculation with 1/9 value of params, if err.
                    saveNameFlagA = saveNameFlagA.rstrip("2")+"3" # 
                    inSMH = list(map(lambda x: x*10, inSMH_0)) 
                    popt_Ani_Freq, poptAni_T, y_A_F, med_npA_F, mo_npA_F, ri_npA_F, fitA_F, isShapeA_freq, isRunA_freq, fitA_T, ri_npA_T, mo_npA_T, med_npA_T, y_npA_T, yori_npA_T, strenRateA = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_A, dataT_A, dicF_A, dataF_A, saveBaseN_I, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqA, searchDir_TimeA, saveNameFlagA, 2, ri_npD_T, ri_med_npD_T, mo_npM_T, med_npM_T)
                    if(bayes_iterCtEx>=5 and (isShapeA_freq==0 or isRunA_freq==0)): # recalculation, if NG data.
                        saveNameFlagA = saveNameFlagA.rstrip("3")+"4" #
                        inSMH = list(map(lambda x: x*11, inSMH_0)) 
                        popt_Ani_Freq, poptAni_T, y_A_F, med_npA_F, mo_npA_F, ri_npA_F, fitA_F, isShapeA_freq, isRunA_freq, fitA_T, ri_npA_T, mo_npA_T, med_npA_T, y_npA_T, yori_npA_T, strenRateA = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_A, dataT_A, dicF_A, dataF_A, saveBaseN_I, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqA, searchDir_TimeA, saveNameFlagA, 2, ri_npD_T, ri_med_npD_T, mo_npM_T, med_npM_T)
                        if(bayes_iterCtEx>=5 and (isShapeA_freq==0 or isRunA_freq==0)): # recalculation, if NG data.
                            saveNameFlagA = saveNameFlagA.rstrip("4")+"5" #
                            inSMH = list(map(lambda x: x*1.055, inSMH_0)) 
                            popt_Ani_Freq, poptAni_T, y_A_F, med_npA_F, mo_npA_F, ri_npA_F, fitA_F, isShapeA_freq, isRunA_freq, fitA_T, ri_npA_T, mo_npA_T, med_npA_T, y_npA_T, yori_npA_T, strenRateA = execFitting_Time_and_Freq(inSMH[0], inSMH[1], inSMH[2], cate, dicT_A, dataT_A, dicF_A, dataF_A, saveBaseN_I, bayes_iterCt, bayes_iterCtEx, dataSplitCnt, mdNo, searchDir_FreqA, searchDir_TimeA, saveNameFlagA,3, ri_npD_T, ri_med_npD_T, mo_npM_T, med_npM_T)
        except IOError as ex:
            cate = cate + " " + str(type(ex)) + " " + str(ex)
            keepProcessResult(0, cate, searchDir_FreqA, "", "", 0, 0, [], "", 0, 0, 0, 0) # keep process data
            isAniexist = 0

    # For getting comp rate
    isCompRate = 0
    mob_mape_freq   = -1
    rigid_dq_freq   = -1
    inter_mape_freq = -1
    inter_dq_freq   = -1
    aniTotal_freq   = -1
    mob_mapeRate   = -1
    rigid_dqRate   = -1
    inter_mapeRate = -1
    inter_dqRate   = -1
    aniTotalRate   = -1
    op_dqRate    = -1
    op_mapeRate  = -1
    op_dqRate2   = -1
    op_mapeRate2 = -1
    rateXmape = -1
    rateYdq   = -1
    ckData = ""
    if(isDQexist==0):
        ckData += "no-DQdata"+"\n"
    if(isMAPEexist==0):
        ckData += "no-MAPEdata"+"\n"
    if(isAniexist==0):
        ckData += "no-Anisotropydata"+"\n"
    if(isShapeD_freq==0):
        ckData += "NG-DQshape"+"\n"
    if(isShapeM_freq==0):
        ckData += "NG-MAPEshape"+"\n"
    if(isShapeA_freq==0):
        ckData += "NG-Anisotropyshape"+"\n"
    if(isRunD_freq==0):
        ckData += "NG-DQcalc"+"\n"
    if(isRunM_freq==0):
        ckData += "NG-MAPEcalc"+"\n"

    if (isDQexist==1 and isMAPEexist==1 and isAniexist==1 and isShapeD_freq==1 and isShapeM_freq==1 and isShapeA_freq==1 and isRunD_freq==1 and isRunM_freq==1):
    #if (1): 
        # adj intensity [Freq]
        dmrate = np.amax(fitD_T)/np.amax(fitM_T) # fit-DQ(rigid)/fit-mape(mobile) #
        dmrateA_D = np.amax(fitA_T)/2/np.amax(fitD_T)
        dmrateA_M = np.amax(fitA_T)/2/np.amax(fitM_T)
        # Freq adj inten to half of Ani
        mo_npM_freq   = mo_npM_freq*dmrateA_M 
        med_np_M_freq = med_np_M_freq*dmrateA_M
        ri_npD_freq   = ri_npD_freq*dmrateA_D  
        med_np_D_freq = med_np_D_freq*dmrateA_D
        fit_D_freq    = fit_D_freq*dmrateA_D   
        fit_M_freq    = fit_M_freq*dmrateA_M   

        # Calc Comp Rate [Freq-inten]
        freqRD_List = calcCompRate2D(isCalBayFor2D, med_npA_F, mo_npA_F, ri_npA_F, fitA_F, mo_npM_freq, ri_npD_freq, med_np_M_freq, med_np_D_freq, saveDirName, filePrefix, saveNameFlag, bayes_repCnt, bayes_iterCt, searchDir_TimeD, searchDir_TimeA, y_npA_T, 1, ckData, isDQexist, isMAPEexist, isAniexist, isShapeD_freq, isShapeM_freq, isRunD_freq, isRunM_freq)

        # adj intensity [Time]        
        mo_npM_T     = mo_npM_T*    np.amax(mo_npM_freq)/np.amax(mo_npM_T)       # unify inten bet. Time & Freq (adj Time to Freq)
        med_npM_T    = med_npM_T*   np.amax(med_np_M_freq)/np.amax(med_npM_T)    # unify inten bet. Time & Freq (adj Time to Freq)
        ri_npD_T     = ri_npD_T*    np.amax(ri_npD_freq)/np.amax(ri_npD_T)       # unify inten bet. Time & Freq (adj Time to Freq)
        ri_med_npD_T = ri_med_npD_T*np.amax(med_np_D_freq)/np.amax(ri_med_npD_T) # unify inten bet. Time & Freq (adj Time to Freq)
        
        # Calc Comp Rate [Time-inten]
        timeRD_List = calcCompRate2D(isCalBayFor2D, med_npA_T, mo_npA_T, ri_npA_T, fitA_T, mo_npM_T, ri_npD_T, med_npM_T, ri_med_npD_T, saveDirName, filePrefix, saveNameFlag, bayes_repCnt, bayes_iterCt, searchDir_TimeD, searchDir_TimeA, y_npA_T, 0, ckData, isDQexist, isMAPEexist, isAniexist, isShapeD_freq, isShapeM_freq, isRunD_freq, isRunM_freq)

        # Calc Comp Rate [3D: time-freq-inten]
        treeDRD_List, ckDataStr, dominanceStr = calcCompRate3D(isCalBayFor2D, saveDirName, filePrefix, searchDir_TimeD, searchDir_TimeA, med_np_M_freq, mo_npM_freq, ri_npD_freq, med_np_D_freq, fitA_F, mo_npA_F, med_npA_F, ri_npA_F, fitM_T.shape[0], mo_npM_T, med_npM_T, ri_med_npD_T, ri_npD_T, fitA_T, mo_npA_T, med_npA_T, ri_npA_T, bayes_repCnt, bayes_iterCt, dmrateA_D,dmrateA_M)
        # show total result
        showResultSummay(treeDRD_List, freqRD_List, timeRD_List, bayes_iterCtEx, saveDirName, filePrefix, searchDir_TimeA, searchDir_TimeD, ckDataStr, dominanceStr, ckData)
    else:
        showResultSummay(["ERROR","-","-","-","-","-"], ["-","-","-","-","-","-"], ["-","-","-","-","-","-"], bayes_iterCtEx, saveDirName, filePrefix, searchDir_TimeA, searchDir_TimeD, "", "-[times]", ckData)

    print ("Program is finished!")


##############################################
# Calc. Comp-rate 
# dataF_A & dataT_A is global 
##############################################
def calcCompRate3D(isCalBayFor2D, saveDirName, filePrefix, searchDir_TimeD, searchDir_TimeA, med_np_M_freq, mo_npM_freq, ri_npD_freq, med_np_D_freq, fitA_F, mo_npA_F, med_npA_F, ri_npA_F, timeAxCnt, mo_npM_T, med_npM_T, med_npD_T, ri_npD_T, fitA_T, mo_npA_T, med_npA_T, ri_npA_T, bayes_repCnt, bayes_iterCt, dmrateA_D,dmrateA_M):

    # chek data & centering: if totMin < 0, this rate's calculation leads to a mistake.
    med_np_M_freq, mo_npM_freq, ri_npD_freq, med_np_D_freq, mo_npA_F, med_npA_F, ri_npA_F, mid_xList, mid_xNP, totMin = supMR.centeringFreqNPs6(med_np_M_freq, mo_npM_freq, ri_npD_freq, med_np_D_freq, mo_npA_F, med_npA_F, ri_npA_F)
    ckDataStr = ""
    if(totMin<0):
        ckDataStr = "[Caution] This calculation is unsuccessful."

    # [Intensity] 
    mob_Stren_max_M  = np.amax(mo_npM_freq)   # freq-mob (same as np.amax(mo_npM_T))
    mobi_Stren_max_M = np.amax(med_np_M_freq) # freq-mob-inter (same as np.amax(med_npM_T))
    rig_Stren_max_D  = np.amax(ri_npD_freq)   # freq-rig (same as np.amax(ri_npD_T))
    rigi_Stren_max_D = np.amax(med_np_D_freq) # freq-rig-inter
    mob_Stren_max_A  = np.amax(mo_npA_F)      # freq-Ani_mob
    medi_Stren_max_A = np.amax(med_npA_F)     # freq-Ani_med
    rigi_Stren_max_A = np.amax(ri_npA_F)      # freq-Ani_rig
    fit_stren_max_A  = np.amax(fitA_F)

    # [T2] - from MAPE/DQ
    t2_mobT, stDmy, endDmy     = supMR.getSpectrumT2(mo_npM_T)  # T2-mob_T
    t2_mobT_med, stDmy, endDmy = supMR.getSpectrumT2(med_npM_T) # T2-mob_inter_T
    t2_riT, stDmy, endDmy      = supMR.getSpectrumT2(ri_npD_T)  # T2-rid_T
    t2_riT_med, stDmy, endDmy  = supMR.getSpectrumT2(med_npD_T) # T2-rid_inter_T

    # [T2] - from Ani.
    t2_AmobT, stDmy, endDmy = supMR.getSpectrumT2(mo_npA_T)  # T2-mob-Ani
    t2_AmedT, stDmy, endDmy = supMR.getSpectrumT2(med_npA_T) # T2-med-Ani
    t2_AridT, stDmy, endDmy = supMR.getSpectrumT2(ri_npA_T)  # T2-rid-Ani
    t2_AfitT, stDmy, endDmy = supMR.getSpectrumT2(fitA_T)  # T2-fit-Ani

    # [freq Width] - from MAPE/DQ
    wid_mobF, stDmy, endDmy  = supMR.getSpectrumWidth(mo_npM_freq, -5, fitA_F, -1)      # wid-mob_F 
    wid_mobF_med, stDmy, endDmy = supMR.getSpectrumWidth(med_np_M_freq, -5, fitA_F, -1) # wid-mob_inter_F
    wid_ridF_med, stDmy, endDmy = supMR.getSpectrumWidth(med_np_D_freq, -5, fitA_F, -1) # wid-rid_inter_F
    wid_ridF, stDmy, endDmy  = supMR.getSpectrumWidth(ri_npD_freq, -5, fitA_F, -1)      # wid-rid_F

    # [freq Width] - from Ani.
    wid_AmobF, stDmy, endDmy  = supMR.getSpectrumWidth(mo_npA_F, -5, fitA_F, -1)        # wid-mob-Ani
    wid_AmedF, stDmy, endDmy  = supMR.getSpectrumWidth(med_npA_F, -5, fitA_F, -1)       # wid-med-Ani
    wid_AridF, stDmy, endDmy  = supMR.getSpectrumWidth(ri_npA_F, -5, fitA_F, -1)        # wid-rid-Ani
    wid_AfitF, stDmy, endDmy  = supMR.getSpectrumWidth(fitA_F, -5, fitA_F, -1)          # wid-fitF-Ani

    # freq width - adj.
    val_baseW = 40/wid_AfitF
    wid_mobF = wid_mobF*val_baseW
    wid_ridF = wid_ridF*val_baseW
    wid_mobF_med = wid_mobF_med*val_baseW
    wid_ridF_med = wid_ridF_med*val_baseW
    wid_AmobF = wid_AmobF*val_baseW
    wid_AmedF = wid_AmedF*val_baseW
    wid_AridF = wid_AridF*val_baseW
    wid_AfitF = wid_AfitF*val_baseW

    # T2 -adj.
    val_baseT = 0.0002
    t2_mobT     = t2_mobT*val_baseT/timeAxCnt
    t2_riT      = t2_riT*val_baseT/timeAxCnt
    t2_mobT_med = t2_mobT_med*val_baseT/timeAxCnt
    t2_riT_med  = t2_riT_med*val_baseT/timeAxCnt
    t2_AmobT    = t2_AmobT*val_baseT/timeAxCnt
    t2_AmedT    = t2_AmedT*val_baseT/timeAxCnt
    t2_AridT    = t2_AridT*val_baseT/timeAxCnt
    t2_AfitT    = t2_AfitT*val_baseT/timeAxCnt

    # Weibull coefficient
    W1=1
    W2=2
    W3Mmed=1.5
    W3Rmed=1.5

    # format of matrix
    #xMat = np.linspace(-200, 200, timeAxCnt)
    xMat = np.linspace(-200, 200, 400)
    yMat = np.linspace(0, 0.001, 100)
    xMat, yMat = np.meshgrid(xMat, yMat)

    # 3d-matrix from MAPE,DQ  ----mobile:Lorentz function/T2 relax.,  rigid: Gaussian function/T2 relax., intermed: (LorentzFunction+GaussianFunction)/relax.
    z_mob_3d  = ((mob_Stren_max_M*wid_mobF**2/(xMat**2+wid_mobF**2))/(np.exp(1/W1*(yMat/t2_mobT)**W1)))
    z_mobi_3d = (mobi_Stren_max_M*(np.exp(-(xMat/wid_mobF_med)**2))+(wid_mobF_med**2/(xMat**2+wid_mobF_med**2)))/((np.exp(1/W3Mmed*(yMat/t2_mobT_med*2)**W3Mmed)))
    z_ridi_3d = (rigi_Stren_max_D*(np.exp(-(xMat/wid_ridF_med)**2))+(wid_ridF_med**2/(xMat**2+wid_ridF_med**2)))/((np.exp(1/W3Rmed*(yMat/t2_riT_med*2)**W3Rmed)))
    z_rid_3d  = (rig_Stren_max_D*np.exp(-(xMat/wid_ridF)**2)/(np.exp(1/W2*(yMat/t2_riT)**W2)))

    # 3d-matrix from Ani
    zA_mob_3d  = ((mob_Stren_max_A*wid_AmobF**2/(xMat**2+wid_AmobF**2))/(np.exp(1/W1*(yMat/t2_AmobT)**W1)))
    zA_med_3d  = (medi_Stren_max_A*(np.exp(-(xMat/wid_AmedF)**2))+(wid_AmedF**2/(xMat**2+wid_AmedF**2)))/((np.exp(1/W3Mmed*(yMat/t2_AmedT*2)**W3Mmed)))
    zA_rid_3d  = (rigi_Stren_max_A*np.exp(-(xMat/wid_AridF)**2)/(np.exp(1/W2*(yMat/t2_AridT)**W2)))
    zA_all     = (zA_mob_3d + zA_med_3d + zA_rid_3d)

    # ----[Only Ani data]----
    tot_3dplot_mobA  = int(np.sum(zA_mob_3d)) # mob_A
    tot_3dplot_medA  = int(np.sum(zA_med_3d)) # med_A
    tot_3dplot_ridA  = int(np.sum(zA_rid_3d)) # rigid_A
    mob_ARate10, med_ARate10, rig_ARate10, minval = supMR.getCompRateAni(tot_3dplot_mobA, tot_3dplot_medA, tot_3dplot_ridA) # Unify decimal point
    if(minval<0): # if min is<0, use max data
        mob_ARate10, med_ARate10, rig_ARate10, minval = supMR.getCompRateAni(np.amax(mo_npA_F), np.amax(med_npA_F), np.amax(ri_npA_F))


    # comp rate with volume
    tot_3dplot_mob  = int(np.sum(z_mob_3d))  # mob
    tot_3dplot_mobi = int(np.sum(z_mobi_3d)) # mob-med
    tot_3dplot_rid  = int(np.sum(z_rid_3d))  # rigid
    tot_3dplot_ridi = int(np.sum(z_ridi_3d)) # rigid-med
    tot_3dplot_all  = int(np.sum(zA_all))    # ani-all

    # ---- rigid-mobile ratio based comp rate with bayesOpt (3D) ----#
    opt_evaluate2, op_dqRate2, op_mapeRate2, keyRate, dominanceStr = getcomprateWzBayesOpt(bayes_repCnt, bayes_iterCt, zA_all, z_mob_3d, z_mobi_3d, z_rid_3d, z_ridi_3d)
    op_dqRate2   = round(op_dqRate2*10,3)
    op_mapeRate2 = round(op_mapeRate2*10,3)

    # ---- spectrum based 4 comp ratios with bayesOpt (3D) ----#
    dq_rig_Rate    = float(round(supMR.calcRateRes2(op_dqRate2,tot_3dplot_rid,tot_3dplot_rid,tot_3dplot_ridi),3))
    dq_rigMed_Rate = float(round(supMR.calcRateRes2(op_dqRate2,tot_3dplot_ridi,tot_3dplot_rid,tot_3dplot_ridi),3))
    ma_mob_Rate    = float(round(supMR.calcRateRes2(op_mapeRate2,tot_3dplot_mob,tot_3dplot_mob,tot_3dplot_mobi),3))
    ma_mobMed_Rate = float(round(supMR.calcRateRes2(op_mapeRate2,tot_3dplot_mobi,tot_3dplot_mob,tot_3dplot_mobi),3))

    # Unify decimal point
    dList = [tot_3dplot_mob, tot_3dplot_mobi, tot_3dplot_ridi, tot_3dplot_rid, tot_3dplot_all]
    width_number = len(str(min(dList))) 
    tranNum = width_number-2
    if(tranNum<0):
        devStr = float("1e"+str(tranNum))
    else:
        devStr = float("1e+"+str(tranNum))

    # ---- Comp rate with volume ----#
    mob_mapeRate = supMR.corrNum(round(tot_3dplot_mob/devStr))
    rigid_dqRate = supMR.corrNum(round(tot_3dplot_rid/devStr))
    inter_mapeRate = supMR.corrNum(round(tot_3dplot_mobi/devStr))
    inter_dqRate   = supMR.corrNum(round(tot_3dplot_ridi/devStr))
    aniTotalRate   = supMR.corrNum(round(tot_3dplot_all/devStr))
 
    # volume ratio derived from bayes result
    b2_mob_mape     = supMR.corrNum(supMR.calcRateRes(mob_mapeRate,mob_mapeRate,inter_mapeRate,op_mapeRate2,op_mapeRate2,op_dqRate2))    
    b2_mob_mape_med = supMR.corrNum(supMR.calcRateRes(inter_mapeRate,mob_mapeRate,inter_mapeRate,op_mapeRate2,op_mapeRate2,op_dqRate2))    
    b2_rig_dq       = supMR.corrNum(supMR.calcRateRes(rigid_dqRate, rigid_dqRate, inter_dqRate, op_dqRate2, op_mapeRate2, op_dqRate2))
    b2_rig_dq_med   = supMR.corrNum(supMR.calcRateRes(inter_dqRate,rigid_dqRate,inter_dqRate,op_dqRate2,op_mapeRate2,op_dqRate2))

    # 3D image
    h_ticks = np.linspace(-200, 200, 5)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.set_xticks(h_ticks)
    ax1.set_yticklabels(["0", "", "", "", "", "0.0010", ""])
    ax1.set_title("3D (Frequency, Time)") #
    ax1.set_xlabel("Frequency[kHz]", fontsize=10)
    ax1.set_ylabel("Time[s]", fontsize=10)
    ax1.set_zlabel("Intensity", fontsize=10)

    ax1.plot_surface(xMat, yMat, zA_all, label="Ani", color="plum",alpha=0.3) 
    ax1.plot_surface(xMat, yMat, z_mob_3d, label="mobile", color="blue",alpha=0.3)
    ax1.plot_surface(xMat, yMat, z_rid_3d, color="green",alpha=0.3)
    ax1.plot_surface(xMat, yMat, z_mobi_3d, color="orange",alpha=0.3)
    ax1.plot_surface(xMat, yMat, z_ridi_3d, color="red",alpha=0.3)

    ax1.set_xlim(np.min(xMat),np.max(xMat))
    ax1.set_ylim(np.min(yMat),np.max(yMat))

    global saveFiles
    saveName = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeA) + "An_Mo_Ri_freq-time_3D.png"
    fig.savefig(saveName)
    plt.close()
    saveFiles += "Saved: "+saveName + "\n"

    # save result of ratio
    if(isShowImage==1):
        cate = "Ratios"
        expStr = showResultRatio(dominanceStr, ckDataStr, dmrateA_D, dmrateA_M, "", 1, cate, tot_3dplot_mob, tot_3dplot_mobi, tot_3dplot_rid, tot_3dplot_ridi, tot_3dplot_all,  mob_mapeRate, inter_mapeRate, rigid_dqRate, inter_dqRate, aniTotalRate,  "-", "-", op_mapeRate2, op_dqRate2, b2_mob_mape, b2_mob_mape_med, b2_rig_dq, b2_rig_dq_med, 1, 1, 1, 1, 1, 1, 1, mob_ARate10, med_ARate10, rig_ARate10, dq_rig_Rate, dq_rigMed_Rate, ma_mob_Rate, ma_mobMed_Rate)
        saveName = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeA) + "-" + os.path.basename(searchDir_TimeD) + "_Frequency&Time_Res_Ratio.txt"
        f = open(saveName, 'w')
        f.write(expStr)

    if(isGUI==-1):
        return [op_mapeRate2, op_dqRate2, str(ma_mob_Rate)+"", str(ma_mobMed_Rate)+"", str(dq_rig_Rate)+"", str(dq_rigMed_Rate)+"", str(mob_ARate10), str(med_ARate10), str(rig_ARate10)], ckDataStr, dominanceStr
    else:
        return [op_mapeRate2, op_dqRate2, str(ma_mob_Rate)+"", str(ma_mobMed_Rate)+"", str(dq_rig_Rate)+"", str(dq_rigMed_Rate)+""] ,ckDataStr, dominanceStr


##############################################
# Calc comp rate
# (dataF_A & dataT_A are global val)
##############################################
def calcCompRate2D(isCalBayFor2D, med_npA, mo_npA, ri_npA, fitA, mo_npM, ri_npD, med_np_M, med_np_D, saveDirName, filePrefix, saveNameFlag, bayes_repCnt, bayes_iterCt, searchDir_TimeD, searchDir_TimeA, y_npA_T, isFreg, ckData, isDQexist, isMAPEexist, isAniexist, isShapeD_freq, isShapeM_freq, isRunD_freq, isRunM_freq):
    if(isFreg==1): # Freq.
        ft_str = "_freq"
        g_title = "Frequency (composite)"
        g_xlabel = "Frequency[kHz]"
    else:          # Time
        ft_str = "_time"
        g_title = "Time (composite)"
        g_xlabel = "Time[s]"

    aniData = fitA
    xList = list(range(aniData.shape[0]))
    xNP = np.array(xList)
    xListAA = list(range(dataF_A.shape[0]))
    xNPAA = np.array(xListAA)
    isCompRate = 1

    # Comp Rate with max of intensity
    mob_mape_freq   = int(np.amax(mo_npM))   # Mobole-MAPE 
    rigid_dq_freq   = int(np.amax(ri_npD))   # Rigid-DQ
    inter_mape_freq = int(np.amax(med_np_M)) # Interphase-MAPE
    inter_dq_freq   = int(np.amax(med_np_D)) # Interphase-DQ
    aniTotal_freq   = int(np.amax(aniData))  # fit-Ani(total)
    mob_Ani_freq    = int(np.amax(mo_npA))   # Mobile-Ani
    rigid_Ani_freq = int(np.amax(ri_npA))    # Rigid-Ani
    inter_Ani_freq = int(np.amax(med_npA))   # Interphase-Ani

    # Comp Rate with sum (area)
    mob_mape_freq_sum   = int(np.sum(mo_npM))   # Mobole-MAPE 
    rigid_dq_freq_sum   = int(np.sum(ri_npD))   # Rigid-DQ
    inter_mape_freq_sum = int(np.sum(med_np_M)) # Interphase-MAPE
    inter_dq_freq_sum   = int(np.sum(med_np_D)) # Interphase-DQ
    aniTotal_freq_sum   = int(np.sum(aniData))  # fit-Ani(total)

    # each area (integral val)
    integralA  = int(integrate.trapz(aniData, xNP))  # 
    integralM  = int(integrate.trapz(mo_npM, xNP))   # 
    integralIM = int(integrate.trapz(med_np_M, xNP)) # 
    try:
        integralR = int(integrate.trapz(ri_npD, xNP)) 
    except ValueError as ex:
        print ("ValueError in integral:", ex) #  check possibility of no data
    integralIR = int(integrate.trapz(med_np_D, xNP)) # 

    integralA_med  = int(integrate.trapz(med_npA, xNP))  # 
    integralA_mob  = int(integrate.trapz(mo_npA, xNP))  # 
    integralA_rig  = int(integrate.trapz(ri_npA, xNP))  # 

    #------ TimeD data: Creating graph (st) ------#
    df = pd.DataFrame({'x':xNP, 'mo_npM':mo_npM, 'ri_npD':ri_npD, 'med_np_M':med_np_M, 'med_np_D':med_np_D, 'aniData':aniData})

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot('x', 'mo_npM', data=df, label='mobile', marker='o', markersize=5)
    ax.plot('x', 'med_np_M', data=df, label='mobile-intermediate', marker='o', markersize=5)
    ax.plot('x', 'ri_npD', data=df, label='rigid', marker='o', markersize=5)
    ax.plot('x', 'med_np_D', data=df, label='rigid-intermediate', marker='o', markersize=5)
    ax.plot('x', 'aniData', data=df, label='anisotropy', marker='o', markersize=5)
    ax.legend(fontsize=10)
    #ax.legend()
    ax.set_title(g_title)
    ax.set_xticklabels(["", "0", "", "", "", "", "", "", "", "0.0010"])
    ax.set_xlabel(g_xlabel, fontsize=10)
    ax.set_ylabel('Intensity', fontsize=10)

    global saveFiles
    if(isFreg==1): # freq.
        termStr = ft_str+"-noedit.png"
    else:
        termStr = ft_str+".png"
        saveName = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeA) + "An_Mo_Ri"+termStr
        fig.savefig(saveName)
        plt.close()
        saveFiles += "Saved: "+saveName + "\n"
    
    #------ TimeD data: Creating graph (end) ------#

    #------ FreqD data: Transfer to centering data & creating graphs (st) ------#
    midE_np_A, midE_np_M, midE_np_IM, midE_np_R, midE_np_IR, mid_xList, mid_xNP = supMR.centeringFreqNPs(aniData, mo_npM, med_np_M, ri_npD, med_np_D)
    if(isFreg==1): # 
        t_rate = 400/mid_xNP.shape[0]
        df = pd.DataFrame({'x': mid_xNP, 'midE_np_M': midE_np_M, 'midE_np_R': midE_np_R, 'midE_np_IM': midE_np_IM, 'midE_np_IR': midE_np_IR, 'midE_np_A': midE_np_A})
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticklabels(["", "-200", "", "", "", "200"])
        ax.plot('x', 'midE_np_M', data=df, label='mobile', marker='o', markersize=3)
        ax.plot('x', 'midE_np_IM', data=df, label='mobile-intermediate', marker='o', markersize=3)
        ax.plot('x', 'midE_np_R', data=df, label='rigid', marker='o', markersize=3)
        ax.plot('x', 'midE_np_IR', data=df, label='rigid-intermediate', marker='o', markersize=3)
        ax.plot('x', 'midE_np_A', data=df, label='anisotropy', marker='o', markersize=3) # 
        ax.legend()
        g_title = "Frequency (composite)"
        g_xlabel = "Frequency[kHz]"
        ax.set_title(g_title)
        ax.set_xlabel(g_xlabel, fontsize=10)
        ax.set_ylabel('Intensity', fontsize=10)

        saveName = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeA) + "An_Mo_Ri"+ft_str+".png"
        fig.savefig(saveName)
        plt.close()
        saveFiles += "Saved: "+saveName + "\n"
    # ------------ FreqD data: Transfer to centering data & creating graphs (end) ------------#

    # ---- Optimization ----#
    greComDiv  = np.gcd.reduce([integralM, rigid_dq_freq, inter_mape_freq, inter_dq_freq, integralA]) # greatest common devisor
    rateXmape, rateYdq = supMR.getOptiAni(mob_mape_freq, rigid_dq_freq, inter_mape_freq, inter_dq_freq, aniTotal_freq, mo_npM, ri_npD, med_np_M, med_np_D, aniData ,xNP, searchDir_TimeA, saveDirName, filePrefix, saveNameFlag, integralM, integralIM, integralR, integralIR, integralA) 

    # ---- Comp rate ----# 
    useV = 3 # 1:max of intensity, 2: sum of intensity(area data) 3: integral (area data)
    if (useV == 1):
        fortotRate_A     = aniTotal_freq
        fortotRate_M     = mob_mape_freq
        fortotRate_M_med = inter_mape_freq
        fortotRate_R     = rigid_dq_freq
        fortotRate_R_med = inter_dq_freq
    elif (useV == 2):
        fortotRate_A     = aniTotal_freq_sum
        fortotRate_M     = mob_mape_freq_sum
        fortotRate_M_med = inter_mape_freq_sum
        fortotRate_R     = rigid_dq_freq_sum
        fortotRate_R_med = inter_dq_freq_sum
    elif (useV == 3):
        fortotRate_A     = integralA
        fortotRate_M     = integralM
        fortotRate_M_med = integralIM
        fortotRate_R     = integralR
        fortotRate_R_med = integralIR

    dList = [abs(fortotRate_M), abs(fortotRate_R), abs(fortotRate_M_med), abs(fortotRate_R_med), abs(fortotRate_A)]
    width_number = len(str(min(dList))) 
    tranNum = width_number-2
    if(tranNum<0):
        devStr = float("1e"+str(tranNum))
    else:
        devStr = float("1e+"+str(tranNum))
    mob_mapeRate = supMR.corrNum(round(fortotRate_M/devStr))
    rigid_dqRate = supMR.corrNum(round(fortotRate_R/devStr))
    inter_mapeRate = supMR.corrNum(round(fortotRate_M_med/devStr))
    inter_dqRate   = supMR.corrNum(round(fortotRate_R_med/devStr))
    aniTotalRate   = supMR.corrNum(round(fortotRate_A/devStr))


    # ----[Only Ani data]---- Unify decimal point
    mob_ARate10, med_ARate10, rig_ARate10, minval = supMR.getCompRateAni(integralA_mob, integralA_med, integralA_rig)
    if(minval<0):
        mob_ARate10, med_ARate10, rig_ARate10, minval = supMR.getCompRateAni(mob_Ani_freq, inter_Ani_freq, rigid_Ani_freq)

    # ---- comp. rate with bayesOpt bayes (with centering data) ----#
    dominanceStr = "-"
    if(isCalBayFor2D==1):
        opt_evaluate2, op_dqRate2, op_mapeRate2, keyRate, dominanceStr = getcomprateWzBayesOpt(bayes_repCnt, bayes_iterCt, midE_np_A, midE_np_M, midE_np_IM, midE_np_R, midE_np_IR)
        op_dqRate2   = round(op_dqRate2*10,3)
        op_mapeRate2 = round(op_mapeRate2*10,3)

        # calc rate with bayesOpt
        dq_rig_Rate    = float(round(supMR.calcRateRes2(op_mapeRate2,  mob_mapeRate,   mob_mapeRate, inter_mapeRate),3))
        dq_rigMed_Rate = float(round(supMR.calcRateRes2(op_mapeRate2,  inter_mapeRate, mob_mapeRate, inter_mapeRate),3))
        ma_mob_Rate    = float(round(supMR.calcRateRes2(op_dqRate2,rigid_dqRate, rigid_dqRate, inter_dqRate),3))
        ma_mobMed_Rate = float(round(supMR.calcRateRes2(op_dqRate2,inter_dqRate, rigid_dqRate, inter_dqRate),3))
    else:
        op_mapeRate2 = "-"
        op_dqRate2   = "-"
        dq_rig_Rate  = "-"
        dq_rigMed_Rate = "-"
        ma_mob_Rate = "-"
        ma_mobMed_Rate = "-"

    if(isShowImage==1):
        # save result of ratio
        cate = "Ratios"
        expStr = showResultRatio(dominanceStr, "", "", "", ckData, isCompRate, cate, fortotRate_M, fortotRate_M_med, fortotRate_R, fortotRate_R_med, fortotRate_A, mob_mapeRate, inter_mapeRate, rigid_dqRate, inter_dqRate, aniTotalRate, rateXmape, rateYdq, op_mapeRate2, op_dqRate2, "-", "-", "-", "-", isDQexist, isMAPEexist, isAniexist, isShapeD_freq, isShapeM_freq, isRunD_freq, isRunM_freq, mob_ARate10, med_ARate10, rig_ARate10, dq_rig_Rate, dq_rigMed_Rate, ma_mob_Rate, ma_mobMed_Rate)
        saveName = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeA) + "-" + os.path.basename(searchDir_TimeD) + ft_str + "Res_Ratio.txt"
        f = open(saveName, 'w')
        f.write(expStr)

    # save process data on first time, becase outputProcess is global
    if(isShowImage==1):
        saveName = saveDirName + "/" + filePrefix + os.path.basename(searchDir_TimeA) + "-" + os.path.basename(searchDir_TimeD) + "_Res_Process.txt"
        f = open(saveName, 'w')
        f.write(outputProcess)

    if(isGUI==-1):
        return [op_mapeRate2, op_dqRate2, str(ma_mob_Rate)+"", str(ma_mobMed_Rate)+"", str(dq_rig_Rate)+"", str(dq_rigMed_Rate)+"", str(mob_ARate10), str(med_ARate10), str(rig_ARate10)]
    else:
        return [op_mapeRate2, op_dqRate2, str(ma_mob_Rate)+"", str(ma_mobMed_Rate)+"", str(dq_rig_Rate)+"", str(dq_rigMed_Rate)+""]
   

##############################################
# Calc comp rate with BayesOpt
##############################################
def getcomprateWzBayesOpt(bayes_repCnt, bayes_iterCt, zA_all, z_mob_3d, z_mobi_3d, z_rid_3d, z_ridi_3d):
    dominanceMobCnt = 0 
    dominanceRigCnt = 0
    rateDicMob = {}
    rateDicRig = {}
    dominanceStr = "-"
    while (1):
        if(dominanceMobCnt==bayes_repCnt):
            tmpList = []
            for keyOrd in rateDicMob:
                tmpList.append(keyOrd)
            tmpList.sort()
            if(bayes_repCnt>=2):
                keyRate = tmpList[int(math.ceil(bayes_repCnt/2))] #
            else:
                keyRate = tmpList[0]
            opt_evaluate2 = rateDicMob[keyRate][0]
            op_dqRate2 = rateDicMob[keyRate][1]
            op_mapeRate2 = rateDicMob[keyRate][2]
            domtimes = dominanceMobCnt + dominanceRigCnt
            dominanceStr = str(dominanceMobCnt)+"/"+str(domtimes) + "[times]"

            break
        elif(dominanceRigCnt==bayes_repCnt):
            tmpList = []
            for keyOrd in rateDicRig:
                tmpList.append(keyOrd)
            tmpList.sort()

            if(bayes_repCnt>=2):
                keyRate = tmpList[int(math.ceil(bayes_repCnt/2))] #
            else:
                keyRate = tmpList[0]            
            opt_evaluate2 = rateDicRig[keyRate][0]
            op_dqRate2 = rateDicRig[keyRate][1]
            op_mapeRate2 = rateDicRig[keyRate][2]

            domtimes = dominanceMobCnt + dominanceRigCnt
            dominanceStr = str(dominanceRigCnt)+"/"+str(domtimes) + "[times]"
            break

        # bayesOptim
        opt_evaluate2, op_dqRate2, op_mapeRate2 = bayesOpt_Rate_Edi(bayes_iterCt, zA_all, z_mob_3d, z_mobi_3d, z_rid_3d, z_ridi_3d)
        if(op_dqRate2==0 or op_mapeRate2==0):
            continue
        if (op_dqRate2 < op_mapeRate2):
            dominanceMobCnt += 1
            keyRate = op_mapeRate2/op_dqRate2
            rateDicMob[keyRate] = [opt_evaluate2, op_dqRate2, op_mapeRate2]
        else:
            dominanceRigCnt += 1
            keyRate = op_dqRate2/op_mapeRate2
            rateDicRig[keyRate] = [opt_evaluate2, op_dqRate2, op_mapeRate2]
    return opt_evaluate2, op_dqRate2, op_mapeRate2, keyRate, dominanceStr


##############################################
# Bayesian-optimization (X*DQ-Y*MAPE --> Ani)
# using central edited data
##############################################
def bayesOpt_Rate_Edi(bayes_iterCt, midE_np_A, midE_np_M, midE_np_IM, midE_np_R, midE_np_IR):
    # Blackbox func.
    def opt_rate_edi(rigiRate, mobRate): # 
        # midE_np_A, (midE_np_R, midE_np_IR), (midE_np_M, midE_np_IM)
        dataF_DM_tmp =  np.add(np.add(rigiRate*midE_np_R, rigiRate*midE_np_IR), np.add(mobRate*midE_np_M, mobRate*midE_np_IM))
        rmseEle = -1*(np.sqrt(mean_squared_error(dataF_DM_tmp, midE_np_A))) # 
        return rmseEle #

    pbounds = {'rigiRate':(0.0, 10), 'mobRate':(0.0, 10)} # DQ-MAPE
    optimizer = BayesianOptimization(f=opt_rate_edi, pbounds=pbounds)
    optimizer.maximize(init_points=5, n_iter=bayes_iterCt, acq='ucb') 

    optiDQ = 0 # dummy ini
    optiMAPE = 0 # dummy ini
    maxRMSE = -1000000 # min val of RMSE (dummy ini)
    rpCnt = 0
    for varDic in optimizer.res:
        op_dqRate   = varDic['params']['rigiRate'] # DQ
        op_mapeRate = varDic['params']['mobRate']  # MAPE
        op_tar = varDic['target']

        if(maxRMSE<op_tar or rpCnt ==0) :
            if(round(op_dqRate,4)<0.1 or round(op_dqRate, 4)>9.9 or round(op_mapeRate, 4)<0.1 or round(op_mapeRate, 4)>9.9):
                rpCnt += 1
                continue
            maxRMSE  = float(round(op_tar, 4))      # 
            optiDQ   = float(round(op_dqRate, 4))   # 
            optiMAPE = float(round(op_mapeRate, 4)) # 
        rpCnt += 1

    maxRMSE = -1*maxRMSE # return as correct RMSE
    return maxRMSE, optiDQ, optiMAPE #


if __name__ == "__main__":
  main()
  
