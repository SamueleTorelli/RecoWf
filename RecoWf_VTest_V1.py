import pandas as pd
import matplotlib.pyplot as plt
import json
import utilityV2 as utility
import baseline_calV2 as baseline_cal
import numpy as np
from scipy.ndimage import gaussian_filter1d    
from parser import parse_txt_to_dataframe
import os
import argparse
import sys
import pdb


#Load reco parameters
if __name__== '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--inputfile", help="Input dir containing the csv files")
    parser.add_argument("-e", "--eventunmber", help="Output file name",type=int)
    parser.add_argument("-C", "--config", help="Configuration file")
    args = parser.parse_args()
    
    if not args.inputfile or not args.eventunmber or not args.config:
        print("python3.8 RwcoWF_VTest_V1.py -I /path/to/input/file -e eventumber -C configfile")
        sys.exit(1)
    
        
    #Read analysis parameters from file
    with open(args.config) as f:
        # Preprocess to remove comments
        cleaned_content = utility.remove_comments(f.readlines())
        # Load JSON from cleaned content
        params = json.loads(cleaned_content)
        
    inputfile=args.inputfile
    
    if inputfile.endswith(".txt"):
        df = parse_txt_to_dataframe(inputfile)
    else:
        df = pd.read_hdf(inputfile)

    while(df.columns[-1] != "event"):
        df = df.drop(columns=df.columns[-1])
    
    wf = df[df["event"]==args.eventunmber].copy()

    while(wf.columns[-1] != "event"):
        wf = wf.drop(columns=wf.columns[-1])

    ChList = wf.columns[1:-1].tolist()
    print(ChList)

    utility.replace_inf_with_max(wf,ChList)

    if params["filter"]==True:
        for i in range(len(ChList)):
            wf[ChList[i]]=utility.MeanFilter(wf[ChList[i]],10)

    baselines = []
    baselinesRMS=[]

    utility.PlotWfs(wf,[0,0,0,0],params)

    if(params["fourier_filter"]):
        utility.RemoveNoiseFourier(wf,0.1e8)
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
        baselines,baselinesRMS = baseline_cal.mean_baselane(df,params)
        for i in range(len(ChList)):
            wf[ChList[i]]-=baselines[i]
        print(baselines,baselinesRMS)
        

    wf=utility.CreateWfSum(wf,params,baselines,baselinesRMS)

    ChList = wf.columns[1:-1].tolist() 
    
    dic_time_begin={}
    dic_time_length={}
    
    
    for ch in wf.columns[1:-1].tolist():
        dic_time_begin[ch]=[]
        dic_time_length[ch]=[]

            
    df_mast = pd.DataFrame(columns=['event','channel','time','time_len','integral'])
    
    for i in range(len(ChList)):
        time_b,time_l,integral,ampl,npeaks,is_sat = utility.Analyze(wf.copy(),baselinesRMS,ChList,i,params)
        #time_b,time_l,integral,ampl = utility.IntegrateFullWindow(wf.copy(),baselinesRMS,ChList,i,params)
        #print("number of peaks")
        #print(npeaks)
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
        aux['isSat'] = is_sat
        
        df_mast = pd.concat([df_mast,aux],ignore_index=True) 
        
    print(dic_time_begin)
    print(df_mast.head(50))

    print(len(df_mast))
    
    utility.PlotWfsTimestamps(wf,dic_time_begin,dic_time_length,baselinesRMS,params)
    
