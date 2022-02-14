# -*- coding: utf-8 -*-
"""
Copy Right: 2021 June-
RIKEN Institute
auther: A.kurotani, K.hara
"""


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
import json
from pulp import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings



#################################################
# For creating graph -rigid, mobile, intermediate
#################################################
def gen_cmap_name(cols):
    nmax = float(len(cols)-1)
    color_list = []
    for n, c in enumerate(cols):
        color_list.append((n/nmax, c))
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', color_list)
    

#################################################
# Check graph shape in the result of freq fitting
# Return: OK or NG (type1-7)
#################################################
def checkGraphShape_freq(y, fit, med_np, rimo_np0, rimo_np1, x, mdNo, fitMax_time): # <- rimo_np0 is mobile in Ani.
    ckFit = ""
    retMemo = ""

    # R^2
    r_squared = np.corrcoef(y, fit)[0,1]**2
    # location in peak value (rate)
    peakInd_med = np.argmax(med_np)/med_np.shape[0]  # Intermediate
    peakInd_rimo1 = np.argmax(rimo_np1)/med_np.shape[0] # Rigid(DQ)/Mobile(MAPE)
    if(mdNo==3): # Ani.
        peakInd_rimo0 = np.argmax(rimo_np0)/med_np.shape[0] # Rigid

    # [1] R^2(fit and Real) < 0.95
    cutoffR2 = 0.95
    if(mdNo!=1 and r_squared < cutoffR2): # exclude MAPE data, which plots tend to rough but fitting shape is almost good.
        ckFit += "_1"
        retMemo += "[NG] R^2: " + str(r_squared) + "\n"
        

    # [2] max val <= around 0
    alzero = fitMax_time/500*-1
    if(np.amax(med_np) < 0): # Intermediate
        if((np.amax(med_np)-np.amin(med_np))/np.amax(fit)>0.001): # exclude super*2 small data in med (almost NA)
            ckFit += "_2-1"
            retMemo += "[NG] Val of Peak in intermediate ("+str(np.amax(med_np))+"): <= 0"+"\n"
 
    if(np.amax(rimo_np1) <= 0):  # max(Rigid or Mobile) < 0  <-- rimo_np0 is rigid in Ani.
        ckFit += "_2-2"
        retMemo += "[NG] Max(Rigid or Mobile) "+str(np.amax(rimo_np1))+"): <= 0"+"\n"
    
    if(mdNo ==3 and np.amax(rimo_np0) <= 0):  # max(Mobile in Anisotropy) < 0 (in Ani.)
        ckFit += "_2-3"
        retMemo += "[NG] Max(Mobile in Anisotropy) ("+str(np.amax(rimo_np0))+"): <= 0"+"\n"

    
    # [3] value on center position <= 0
    if(med_np[int(med_np.shape[0]/2)] <= 0): # Intermediate
        if((np.amax(med_np)-np.amin(med_np))/np.amax(fit)>0.001): # exclude super*2 small data in med
            ckFit += "_3-1"
            retMemo += "[NG] Meddle of Val in intermediate ("+str(med_np[int(med_np.shape[0]/2)])+"): <= 0"+"\n"
        
    if(rimo_np1[int(rimo_np1.shape[0]/2)] <= 0): # Rigid or Mobile (Mobile in Ani.)
        ckFit += "_3-2"
        retMemo += "[NG] Meddle of Val in rigid or mobile ("+str(rimo_np1[int(rimo_np1.shape[0]/2)])+"): <= 0"+"\n"
            
        
    if(mdNo ==3 and rimo_np0[int(rimo_np0.shape[0]/2)] <= 0): # Mobile (in Ani.)
        ckFit += "_3-3"
        retMemo += "[NG] Meddle of Val in Anisotropy ("+str(rimo_np0[int(rimo_np0.shape[0]/2)])+"): <= 0"+"\n"

        
    # [4] peak location is in out of center(:48-52%)
    if(peakInd_med < 0.47 and 0.53 < peakInd_med): # Intermediate
        ckFit += "_4-1"
        retMemo += "[NG] Peak location ("+str(peakInd_med)+") of intermadiate layer is not center (47%-53%)"+"\n"

    if(peakInd_rimo1 < 0.47 and 0.53 < peakInd_rimo1): # Rigid(DQ)/Mobile(MAPE)
        ckFit += "_4-2"
        retMemo += "[NG] Peak location ("+str(peakInd_rimo1)+") of rigid/mobile layer is not center (47%-53%)"+"\n"
        
    if(mdNo ==3 and peakInd_rimo0 < 0.47 and 0.53 < peakInd_rimo0): # Mobile in Ani.(MAPE)
        ckFit += "_4-3"
        retMemo += "[NG] Peak location ("+str(peakInd_rimo0)+") of mobile layer (Anisotropy) is not center (47%-53%)"+"\n"
        
    # [5] slope is '+' from start to center, and '-' from center to end (in intermediate)
    slopA_med_bf, inceptB = np.polyfit(x[:int(x.shape[0]/2)], med_np[:int(med_np.shape[0]/2)], 1) # Intermediate(before center)
    slopA_med_af, inceptB = np.polyfit(x[int(x.shape[0]/2):], med_np[int(med_np.shape[0]/2):], 1) # Intermediate(after center)
    if(slopA_med_bf<=0 and slopA_med_af>=0):
        ckFit += "_5-1"
        retMemo += "[NG] slop between 0 and middle (Intermediate): "+str(slopA_med_bf)+" > 0"+"\n"
        retMemo += "     & slop between middle and last (Intermediate): "+str(slopA_med_af)+" > 0"+"\n"

    # slope is + from start to center, and - from center to end (in rigid/mobile, rigid-ani)
    slopA_rid_bf, inceptB = np.polyfit(x[:int(x.shape[0]/2)], rimo_np1[:int(rimo_np1.shape[0]/2)], 1) # (before center)
    slopA_rid_af, inceptB = np.polyfit(x[int(x.shape[0]/2):], rimo_np1[int(rimo_np1.shape[0]/2):], 1) # (after center)
    if(slopA_rid_bf<=0 and slopA_rid_af>=0):
        ckFit += "_5-2"
        if(mdNo ==3):
            retMemo += "[NG] slop between start and middle (rigid): "+str(slopA_rid_bf)+" < 0"+"\n"
            retMemo += "     & slop between middle and last (rigid): "+str(slopA_rid_af)+" > 0"+"\n"
        else:
            retMemo += "[NG] slop between start and middle (rigid/mobile): "+str(slopA_rid_bf)+" < 0"+"\n"
            retMemo += "     & slop between middle and last (rigid/mobile): "+str(slopA_rid_af)+" > 0"+"\n"            

    # slope is + from start to center, and - from center to end (in rigid/mobile, rigid-ani)
    if(mdNo ==3):
        slopA_rid_bf, inceptB = np.polyfit(x[:int(x.shape[0]/2)], rimo_np0[:int(rimo_np0.shape[0]/2)], 1) # (before center)
        slopA_rid_af, inceptB = np.polyfit(x[int(x.shape[0]/2):], rimo_np0[int(rimo_np0.shape[0]/2):], 1) # (after center)
        if(slopA_rid_bf<=0 and slopA_rid_af>=0):
            ckFit += "_5-3"
            retMemo += "[NG] slop between start and middle (Ani-mobile): "+str(slopA_rid_bf)+" < 0"+"\n"
            retMemo += "     & slop between middle and last (Ani-mobile): "+str(slopA_rid_af)+" > 0"+"\n"

    # [6] the both terminals of fitting data are within 1/1000 of peak max
    if (fit[0] > np.amax(fit)/10):
        ckFit += "_6-1"
        retMemo += "[NG] the value in term. of left-side ("+str(fit[0])+") is within among the 1/10 of max < 0. (max/1000: " + str(np.amax(fit)/10) + ")" +"\n"

    if (fit[fit.shape[0]-1] > np.amax(fit)/10):
        ckFit += "_6-2"
        retMemo += "[NG] the value in term of right-side ("+str(fit[fit.shape[0]-1])+") is within among the 1/10 of max < 0. (max/1000: " + str(np.amax(fit)/10) + ")" +"\n"

    # existance of two or more peaks (focusing on out of 10% location from center point)
    bf_np = fit[:int(fit.shape[0]/2)] # before center
    af_np = fit[int(fit.shape[0]/2):] # after center
    if ((bf_np.shape[0]-np.argmax(bf_np))/bf_np.shape[0] >= 0.1 and  np.argmax(af_np)/bf_np.shape[0] >=0.1):
        ckFit = "-2Peaks" # 
        retMemo += "[NG] 2 peaks exist\n" # 

    # [7] peak min val < 0
    alzero = fitMax_time*3/100*-1
    if(np.amin(med_np) < alzero): # Intermediate
        ckFit += "_7-1"
        retMemo += "[NG] Val of Peak in intermediate ("+str(np.amin(med_np))+"): is over low (minus)"+"\n"
        #print (327, np.amin(med_np), np.amax(med_np), alzero, fitMax_time)


    if(np.amin(rimo_np1) < alzero):  # min(Rigid or Mobile) < alzero  <-- rimo_np0 is rigid in Ani.
        ckFit += "_7-2"
        retMemo += "[NG] Min(Rigid or Mobile) "+str(np.amin(rimo_np1))+"): is over low (minus)"+"\n"
        #print (332, np.amin(rimo_np1), np.amax(rimo_np1), alzero, fitMax_time)

    if(mdNo ==3 and np.amin(rimo_np0) < alzero):  # min(Mobile in Anisotropy) < alzero (in Ani.)
        ckFit += "_7-3"
        retMemo += "[NG] Min(Mobile in Anisotropy) ("+str(np.amin(rimo_np0))+"): is over (minus)"+"\n"

    # [8] peak max == infinity
    if(np.amax(med_np)==math.inf):
        ckFit += "_8-1"
        retMemo += "[NG] Max(intermediate) "+" is Infinity"+"\n"
        
    if(np.amax(rimo_np1)==math.inf):
        ckFit += "_8-2"
        retMemo += "[NG] Max(Rigid or Mobile) "+" is Infinity"+"\n"
    
    if(mdNo ==3 and np.amax(rimo_np0)==math.inf):
        ckFit += "_8-3"
        retMemo += "[NG] Max(Mobile in Anisotropy) "+" is Infinity"+"\n"

    
    # [9] freq peak check -exclude flat data, etc.
    ret_width, ret_peakSt, ret_peakEnd = getSpectrumWidth(med_np, -50, fit, mdNo)
    if(ret_peakSt>ret_peakEnd or ret_peakSt==-1 or ret_peakEnd==-1):
        ckFit += "_9-1"
        retMemo += "[NG] spectrum shape in intermediate in freq is Unusual"+"\n"

    ret_width, ret_peakSt, ret_peakEnd = getSpectrumWidth(rimo_np1, -50, fit, mdNo)
    if(ret_peakSt>ret_peakEnd or ret_peakSt==-1 or ret_peakEnd==-1):
        ckFit += "_9-2"
        retMemo += "[NG] spectrum shape in rigid or mobile in freq is Unusual"+"\n"
     
    ret_width, ret_peakSt, ret_peakEnd = getSpectrumWidth(rimo_np0, -50, fit, mdNo)
    if(mdNo ==3 and (ret_peakSt>ret_peakEnd or ret_peakSt==-1 or ret_peakEnd==-1)):
        ckFit += "_9-3"
        retMemo += "[NG] spectrum shape in mobile-Anisotropy in freq is Unusual"+"\n"


    # [10] min is far away from baseline
    if(mdNo==3):
        maxInMins = max([np.amin(rimo_np1), np.amin(rimo_np0)]) # ignore intermed. data
        minInMaxs = min([np.amax(rimo_np1), np.amax(rimo_np0)]) # ignore intermed. data
        if(maxInMins>minInMaxs):
            ckFit += "_10-1"
            retMemo += "[NG] maxvalue in minvalues is higher than minvalue in maxvalues"+"\n"

    else:
        maxInMins = max([np.amin(med_np), np.amin(rimo_np1)])
        minInMaxs = min([np.amax(med_np), np.amax(rimo_np1)])
        if(maxInMins>minInMaxs):
            ckFit += "_10-2"
            retMemo += "[NG] maxvalue in minvalues is higher than minvalue in maxvalues"+"\n"
      

    if(retMemo==""):
        retMemo = "[OK]"

    return ckFit, retMemo


##############################################
# get spectrum's width from frequency np_data 
# extruction rule: slope<-50
# Ani-data--> dataNo==3 
##############################################
def getSpectrumWidth(np_freq, slopeCutoff, fit, dataNo):
    indPeak   = np.argmax(np_freq) # peak position
    ckNM_data = np_freq[indPeak:]  # np_data after peak to last
    if(dataNo==1): # if data is MAPE
        ckDataCnt = 12
    else:
        #ckDataCnt = 100
        ckDataCnt = 20
    minWid = ckDataCnt/2

    if(dataNo==3 and (np.amax(np_freq)-np.amin(np_freq))/np.amax(fit)<=0.005): 
        # ignore super small data as less than 0.5% (in Ani) --The porpus in Ani is getting total fitting-npdata.
        ret_width, ret_peakSt, ret_peakEnd = getSpectrumWidth(fit, slopeCutoff, fit, -1)
    else:
        # slope>-5 or intensity<=0
        ret_width   = ckNM_data.shape[0]
        ret_peakSt  = -1
        ret_peakEnd = -1
        maxSlope = 0
        rpCnt = 0
        slopeNP = np.zeros(1) #
        for i in range(ckNM_data.shape[0]-1):
            if(i < ckNM_data.shape[0]-ckDataCnt):
                tmpY_ckNM_data = ckNM_data[i:i+ckDataCnt] 
                x_axi = np.array(range(i,i+ckDataCnt))
                slope,b=np.polyfit(x_axi,tmpY_ckNM_data,1)
                if(rpCnt==0 and slope > slopeCutoff):
                    #slopeCutoff = slope*0.75
                    slopeCutoff = slope-abs(slope*0.25)
                if(slope < maxSlope):
                    maxSlope = slope
                slopeNP = np.append(slopeNP, slope)
                if((slope>slopeCutoff or ckNM_data[i]<=0) and i > minWid):
                    ret_width = (i+1)*2
                    ret_peakSt  = indPeak-i-2
                    ret_peakEnd = indPeak+i+1
                    break
            rpCnt += 1

        # 2nd check
        startCK = -1
        if(ret_peakSt==-1):
            for j in range(slopeNP.shape[0]-1):
                if(j < slopeNP.shape[0]-ckDataCnt):
                    tmpY_slopeNP = slopeNP[j:j+ckDataCnt] # 
                    axiX = np.array(range(j,j+ckDataCnt)) 
                    slope,b=np.polyfit(axiX, tmpY_slopeNP, 1)
                    if(j==0 and slope<0):
                        startCK = 1
                    if(startCK==1 and slope>0):
                        ret_width = (i+1)*2
                        ret_peakSt  = indPeak-i-2
                        ret_peakEnd = indPeak+i+1
                        break

        if(maxSlope>slopeCutoff*1.2):
            print ("1")
            ret_width   = ckNM_data.shape[0]
            ret_peakSt  = -1
            ret_peakEnd = -1
        elif(ret_peakSt<0 and ret_peakEnd > np_freq.shape[0]-1):
            print ("2")
            ret_peakSt  = 0
            ret_peakEnd = np_freq.shape[0]-1
            ret_width   = np_freq.shape[0]
        elif(ret_peakSt>0 and ret_peakEnd > np_freq.shape[0]-1):
            print ("3")
            ret_peakEnd = np_freq.shape[0]-1
            ret_width   = ret_peakEnd-ret_peakSt+1
        elif(ret_peakSt<0 and ret_peakEnd < np_freq.shape[0]-1):
            print ("4", ret_peakSt, ret_peakEnd, np_freq.shape[0])
            ret_peakSt  = 0
            ret_width   = ret_peakEnd-ret_peakSt+1

    return ret_width, ret_peakSt, ret_peakEnd


######################################
# slope and intercept in linear function from 2 points of X, Y 
# input (x1,y1), (x2,y2)
######################################
def makeLinearEquation(x1, y1, x2, y2):
    x_np = np.array([x1, x2])
    y_np = np.array([y1, y2])
    slop, intercept = np.polyfit(x_np, y_np, 1)
    return slop, intercept


##############################################
# -- widen np data --
# npfit: npData
# centPosi: top of peak
# stM: peak start
# restrictNo: half of roop counts for addition
# (1) need st & top position of peak
# (2) need roop counts for st to top
# (3) use the same roop counts for top to end
##############################################
def widenSpecNP(npfit, centPosi, stM, restrictNo):
    totalCnt = 0
    while (totalCnt < restrictNo):
        widToCent = centPosi-stM+1
        for num in range(widToCent):
            if (num==0):
                pCur  = centPosi
                nCur  = centPosi
                pNext = centPosi+1
                nNext = centPosi-1
            else:
                pCur = pNext+1 # 
                nCur = nNext-1 #
                pNext = pCur+1
                nNext = nCur-1
                #print (1471, num, pCur, pNext, (npfit[pCur]+npfit[pNext])/2, npfit[pCur], npfit[pNext])
                npfit = np.insert(npfit, pCur+1, (npfit[pCur]+npfit[pNext])/2)
                npfit = np.insert(npfit, nCur-1, (npfit[nCur]+npfit[nNext])/2)
                npfit = np.delete(npfit, 0)
                npfit = np.delete(npfit, npfit.shape[0]-1) 
            totalCnt+=1
        stM = stM - widToCent
    return npfit


##############################################
# Comp Rate from only Ani-data
##############################################
def getCompRateAni(tot_3dplot_mobA, tot_3dplot_medA, tot_3dplot_ridA):
    dListA = [tot_3dplot_mobA, tot_3dplot_medA, tot_3dplot_ridA]
    width_numberA = len(str(min(dListA)))
    tranNumA = width_numberA-2
    if(tranNumA<0):
        devStrA = float("1e"+str(tranNumA))
    else:
        devStrA = float("1e+"+str(tranNumA))
    # ---- Comp rate with volume ----#
    mob_ARate   = tot_3dplot_mobA/devStrA
    med_ARate   = tot_3dplot_medA/devStrA
    rigid_ARate = tot_3dplot_ridA/devStrA
    totAArate = mob_ARate + med_ARate + rigid_ARate
    #print (322, mob_ARate, totAArate, mob_ARate, mob_ARate, "|", tot_3dplot_mobA,tot_3dplot_medA, tot_3dplot_ridA)
    mob_ARate10 = corrNum(round(mob_ARate/totAArate*10,1))
    med_ARate10 = corrNum(round(med_ARate/totAArate*10,1))
    rig_ARate10 = round(10-mob_ARate10-med_ARate10,1)
    #print ("1384_mobile/intermediate/rigid", mob_ARate10, med_ARate10, rig_ARate10, dListA)
    ckMinL = [mob_ARate10, med_ARate10, rig_ARate10]
    return mob_ARate10, med_ARate10, rig_ARate10, min(ckMinL)


##############################################
# calcRate1
############################################## 
def calcRateRes(rate1, rate2, rate3, rate4, rate5, rate6):
    ret = 0
    if ((rate2 + rate3)*rate4 == 0):
        pass
    elif (rate5+rate6==0):
        pass
    else:
        ret = round(rate1/(rate2 + rate3)*rate4/(rate5+rate6)*100,1)
    return ret

##############################################
# calcRate2
############################################## 
def calcRateRes2(rate1, rate2, rate3, rate4):
    ret = -1
    if(rate3+rate4==0):
        pass
    else:
        if(rate1<0 or rate2<0):
            pass
        else:
            ret = rate1*rate2/(rate3+rate4)*10
    return ret

##############################################
# correct -0 to 0
##############################################    
def corrNum(n_num):
    ret = n_num
    n_numStr = str(n_num)
    if( n_numStr[0]=="-"):
        n_numStr =  n_numStr.replace('-','')
        n_numFl  = float(n_numStr)
        ret = n_numFl
    #if(n_num<0):
    #    pass
    #else:
    #    n_num = abs(n_num)
    return ret

##############################################
# cntering: edit freq-np-peak to center (1)
##############################################
def centeringFreqNPs(aniData, mo_npM, med_np_M, ri_npD, med_np_D):
    # peak location
    mid_ind_A  = np.argmax(aniData)
    mid_ind_M  = np.argmax(mo_npM)
    mid_ind_IM = np.argmax(med_np_M)
    mid_ind_R  = np.argmax(ri_npD)
    mid_ind_IR = np.argmax(med_np_D)
    ranV1 = min(mid_ind_A, mid_ind_M, mid_ind_IM, mid_ind_R, mid_ind_IR) # left-side of graph (smaller)
    ranV2 = aniData.shape[0]-max(mid_ind_A, mid_ind_M, mid_ind_IM, mid_ind_R, mid_ind_IR)-1 # right-side of graph (biger)
    ranV = min(ranV1, ranV2)

    # np matrix
    midE_np_A  = aniData[mid_ind_A-ranV:mid_ind_A+ranV+1]
    midE_np_M  = mo_npM[mid_ind_M-ranV:mid_ind_M+ranV+1]
    midE_np_IM = med_np_M[mid_ind_IM-ranV:mid_ind_IM+ranV+1]
    midE_np_R  = ri_npD[mid_ind_R-ranV:mid_ind_R+ranV+1]
    midE_np_IR = med_np_D[mid_ind_IR-ranV:mid_ind_IR+ranV+1]
    mid_xList = list(range(midE_np_A.shape[0]))
    mid_xNP = np.array(mid_xList)
    return midE_np_A, midE_np_M, midE_np_IM, midE_np_R, midE_np_IR, mid_xList, mid_xNP

############################################################################################
# cntering: edit freq-np-peak to center (2)
# aniData, mo_npM, med_np_M, np_4, np_5, np_6, np_7
# --> med_np_M_freq, mo_npM_freq, ri_npD_freq, med_np_D_freq, mo_npA_F, med_npA_F, ri_npA_F
############################################################################################
def centeringFreqNPs6(aniData, np_2, np_3, np_4, np_5, np_6, np_7):
    w_Ani, stA, endA  = getSpectrumWidth(aniData, -5, aniData, -1) # Ani
    w_np2, st2, end2  = getSpectrumWidth(np_2, -5, aniData, 3)
    w_np3, st3, end3  = getSpectrumWidth(np_3, -5, aniData, 3)
    w_np4, st4, end4  = getSpectrumWidth(np_4, -5, aniData, 3)
    w_np5, st5, end5  = getSpectrumWidth(np_5, -5, aniData, 3)
    w_np6, st6, end6  = getSpectrumWidth(np_6, -5, aniData, 3)
    w_np7, st7, end7  = getSpectrumWidth(np_7, -5, aniData, 3)

    # check min
    totMinPeak = min(np.amax(aniData), np.amax(np_2), np.amax(np_3), np.amax(np_4), np.amax(np_5), np.amax(np_6), np.amax(np_7))

    # peak location
    mid_ind_A = np.argmax(aniData)
    mid_ind_2 = np.argmax(np_2)
    mid_ind_3 = np.argmax(np_3)
    mid_ind_4 = np.argmax(np_4)
    mid_ind_5 = np.argmax(np_5)
    mid_ind_6 = np.argmax(np_6)
    mid_ind_7 = np.argmax(np_7)

    ranV1 = min(mid_ind_A, mid_ind_2, mid_ind_3, mid_ind_4, mid_ind_5, mid_ind_6, mid_ind_7) # leftside of graph (smaller)
    ranV2 = aniData.shape[0]-max(mid_ind_A, mid_ind_2, mid_ind_3, mid_ind_4, mid_ind_5, mid_ind_6, mid_ind_7)-1 # rightside of graph (biger)
    if(totMinPeak<0):
        #ranV = int(aniData.shape[0]/2)
        ranV = stA
    elif(stA <= ranV1 and ranV1 >= endA):
        ranV = min(ranV1, ranV2)   
    else:
        ranV = stA
    
    # np matrix
    mid_np_1 = aniData[mid_ind_A-ranV:mid_ind_A+ranV+1]
    mid_np_2 = makeNPforCent(np_2, mid_ind_2, ranV)
    mid_np_3 = makeNPforCent(np_3, mid_ind_3, ranV)
    mid_np_4 = makeNPforCent(np_4, mid_ind_4, ranV)
    mid_np_5 = makeNPforCent(np_5, mid_ind_5, ranV)
    mid_np_6 = makeNPforCent(np_6, mid_ind_6, ranV)
    mid_np_7 = makeNPforCent(np_7, mid_ind_7, ranV)
    mid_xList = list(range(mid_np_1.shape[0]))
    mid_xNP = np.array(mid_xList)

    return [mid_np_1, mid_np_2, mid_np_3, mid_np_4, mid_np_5, mid_np_6, mid_np_7, mid_xList, mid_xNP, totMinPeak]

##############################################
# make np for getCenter
##############################################
def makeNPforCent(np_x, mid_ind, ranV):
    st = mid_ind-ranV
    en = mid_ind+ranV+1
    if (st<0):
        en = en + (-1*st)
        st = 0
    mid_np = np_x[st:en]
    return mid_np

##############################################
# get spectrum's width from frequency np_data
# extraction: slope<-5 or intensity<=0
##############################################
def getSpectrumT2(np_freq):
    indPeak   = np.argmax(np_freq) # peak
    ckNM_data = np_freq
    ckDataCnt = 5
    preslope = -1e-20

    # slope<-5 or intensity<=0
    ret_width   = ckNM_data.shape[0]
    ret_peakSt  = -1
    ret_peakEnd = -1
    for i in range(ckNM_data.shape[0]-1):
        if(i < ckNM_data.shape[0]-ckDataCnt):
            tmpY_ckNM_data = ckNM_data[i:i+ckDataCnt] # check 100 plots
            x_axi = np.array(range(i,i+ckDataCnt))
            slope,b=np.polyfit(x_axi,tmpY_ckNM_data,1)
            intenCutoff = int(np.amax(np_freq)/100)
            slopeCutoff = -5
            #print (1431, i, tmpY_ckNM_data.shape, x_axi.shape, slope, ckNM_data[i], np.amax(np_freq), slopeCutoff, intenCutoff)
            if(slope>slopeCutoff or ckNM_data[i]<=intenCutoff or (preslope >= slope and i>2)):
                ret_width = i+1
                ret_peakSt  = 0
                ret_peakEnd = ret_width
                #print (1437, i, slope, ret_width, ret_peakSt, ret_peakEnd, ckNM_data[i], slopeCutoff)
                break
            preslope = slope
    return ret_width, ret_peakSt, ret_peakEnd

     
#################################################################
# Optimization with pulp using R^2 for eva.
# aniTotal = x*mob_mape + x*inter_mape + y*inter_dq + y*rigid_dq
#################################################################
def getOptiAni(mob_mape, rigid_dq, inter_mape, inter_dq, aniTotal, midE_np_M, midE_np_R, midE_np_IM, midE_np_IR, midE_np_A ,mid_xNP, searchDir_TimeA, saveDirAndWord, filePrefix, saveNameFlag, integralM, integralIM, integralR, integralIR, integralA):
    #print("L1220", dataF_A.shape, np.argmax(dataF_A), np.argmax(mo_npM), np.argmax(med_np_M), np.argmax(ri_npD), np.argmax(med_np_D))
    prob = pulp.LpProblem(sense=LpMaximize) # 
    x = pulp.LpVariable('x', lowBound=0) # 
    y = pulp.LpVariable('y', lowBound=0) # 

    addNPsA = np.add(midE_np_M, midE_np_IM)
    addNPsB = np.add(midE_np_R, midE_np_IR)

    # R^2
    prob += R2_inPlup(x, y, addNPsA, addNPsB, midE_np_A) # R^2 of Ani & sum(x*MAPE,y*DQ) -- need func. for avoiding err.
    prob += x*mob_mape + x*inter_mape + y*inter_dq + y*rigid_dq == aniTotal      # ani=sum(x*MAPE,y*DQ)
    prob += x*integralM + x*integralIM + y*integralR + y*integralIR == integralA # integral of ani=sum(x*MAPE,y*DQ) 
    status = prob.solve(PULP_CBC_CMD(msg=False))  
    return x.value(), y.value() # x:mape, y:rigid


#################################################################
# pulp-func. of in getOptiAni()
#################################################################
def R2_inPlup(x, y, addNPsA, addNPsB, midE_np_A):
    x_addNPAs = 2*addNPsA
    y_addNPAs = 3*addNPsB
    return np.corrcoef(np.add(x_addNPAs, y_addNPAs), midE_np_A)[0,1]**2
   

