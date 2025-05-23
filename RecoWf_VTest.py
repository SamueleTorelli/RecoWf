import pandas as pd
import matplotlib.pyplot as plt
import json
import utility
import baseline_cal
import numpy as np
from scipy.ndimage import gaussian_filter1d    
import os

    
#Load reco parameters
if __name__== '__main__':
    with open('reco_params.txt') as f:
        params = json.load(f)

    filedir='/Volumes/Elements/Run_16/'
    
    files = os.listdir(filedir)
    
    wf = pd.read_csv(filedir+'/'+files[83],skiprows=15)
    #wf = wf.drop(columns=wf.columns[-2])

    print(wf['TIME'].info())
    
    ChList = wf.columns[1:].tolist()
    print(ChList)
    
    if params["filter"]==True:
        for i in range(len(ChList)):
            wf[ChList[i]]=utility.MeanFilter(wf[ChList[i]],10)

    baselines = []
    baselinesRMS=[]

    utility.PlotWfs(wf,[0,0,0,0],params)
    
    if(params["variable_mean_rms"]):
        sig_free_time,sig_free_rms = utility.FindSignalFreeRegion(wf,params)

        for chindex,t_st in enumerate(sig_free_time):
            baselines.append(wf[ChList[chindex]].iloc[t_st:t_st + params["n_points_pre_wf"]].mean())
    
        for i in range(len(ChList)):
            wf[ChList[i]]-=baselines[i]
        
        baselinesRMS=sig_free_rms
        utility.PlotWfs(wf,sig_free_time,params)
    else:
        baselines,baselinesRMS = baseline_cal.mean_baselane(filedir,params)
        for i in range(len(ChList)):
            wf[ChList[i]]-=baselines[i]
        print(baselines,baselinesRMS)
        
    dic_time_begin={}
    dic_time_length={}
    
    for ch in wf.columns[1:].tolist():
        dic_time_begin[ch]=[]
        dic_time_length[ch]=[]
    
    df_mast = pd.DataFrame(columns=['event','channel','time','time_len','integral'])
    
    for i in range(len(ChList)):
        time_b,time_l,integral,ampl,npeaks = utility.Analyze(wf.copy(),baselinesRMS,ChList,i,params)
        #time_b,time_l,integral,ampl = utility.IntegrateFullWindow(wf.copy(),baselinesRMS,ChList,i,params)
        print("number of peaks")
        print(npeaks)
        event =[]
        channel = []
        for j in range(len(integral)):
            event.append(1)
            channel.append(ChList[i])

        dic_time_begin[ChList[i]]=time_b
        dic_time_length[ChList[i]]=time_l

        aux = pd.DataFrame([])
        aux['event'] = event
        aux['channel'] = channel
        aux['time'] = time_b
        aux['time_len'] = time_l
        aux['integral'] = integral
        aux['ampl'] = ampl

        df_mast = pd.concat([df_mast,aux],ignore_index=True) 
        
    print(dic_time_begin)
    print(df_mast.head(50))
    utility.PlotWfsTimestamps(wf,dic_time_begin,dic_time_length,baselinesRMS,params)
    
