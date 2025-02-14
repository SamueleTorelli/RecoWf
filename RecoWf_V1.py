import pandas as pd
import matplotlib.pyplot as plt
import json
import utilityV2 as utility
import baseline_calV2 as baseline_cal
import numpy as np
from scipy.ndimage import gaussian_filter1d    
import os
import argparse
import sys
    
#Load reco parameters
if __name__== '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--inputfile", help="Input dir containing the csv files")
    parser.add_argument("-O", "--outfile", help="Output file name")
    parser.add_argument("-C", "--config", help="Configuration file")
    args = parser.parse_args()

    if not args.inputfile or not args.outfile or not args.config:
        print("python_x RwcoWF_V1.py -I /path/to/input/file -O path/to/outfile -C configfile.txt")
        sys.exit(1)

    
    #Read analysis parameters from file
    with open(args.config) as f:
        # Preprocess to remove comments
        cleaned_content = utility.remove_comments(f.readlines())
        # Load JSON from cleaned content
        params = json.loads(cleaned_content)

    
    #Initialize output dataframe
    df_mast = pd.DataFrame(columns=['event','channel','time','time_len','integral'])

    baselines = []
    baselinesRMS=[]

    df = pd.read_hdf(args.inputfile) 

    while(df.columns[-1] != "event"):
        df = df.drop(columns=df.columns[-1])
            
    
    if(params["variable_mean_rms"]==False):
        baselines,baselinesRMS = baseline_cal.mean_baselane(df,params)
        
    #Loop all over the events
    eventlist = np.unique(df["event"].to_list())
    
    #for nev in eventlist:
    for nev in eventlist:
        if nev%10==0:
            print(nev)
        
        #Open df
        wf = df[df["event"]==nev].copy()

                
        ChList = wf.columns[1:-1].tolist()
        utility.replace_inf_with_max(wf,ChList)
        
        #If true applies the Mean Filter to each channel
        if params['filter']==True:
            for i in range(len(ChList)):
                wf[ChList[i]]=utility.MeanFilter(wf[ChList[i]],10)

        if(params["fourier_filter"]):
            utility.RemoveNoiseFourier(wf,0.1e8)
        
        if(params["variable_mean_rms"]):
            sig_free_time,sig_free_rms = utility.FindSignalFreeRegion(wf,params)
        
            #baselines = wf[wf.columns[1:]].iloc[30:30 + params["n_points_pre_wf"]].mean().values
            baselines = []
            for chindex,t_st in enumerate(sig_free_time):
                baselines.append(wf[ChList[chindex]].iloc[t_st:t_st + params["n_points_pre_wf"]].mean())
            baselinesRMS=sig_free_rms

        #Subtract baseline and calculate RMS
        for i in range(len(ChList)):
            wf[ChList[i]]-=baselines[i]

        if(params["analyze_sum"]):
            wf=utility.CreateWfSum(wf,params,baselines,baselinesRMS) 
        ChList = wf.columns[1:-1].tolist()
        
        dic_time_begin={}
        dic_time_length={}
    
        for ch in ChList:
            dic_time_begin[ch]=[]
            dic_time_length[ch]=[]    
        
        for i in range(len(ChList)):
            #Analyze the waveform
            if( not params['full_window']):
                time_b,time_l,integral,ampl,npeaks,is_sat = utility.Analyze(wf.copy(),baselinesRMS,ChList,i,params)
            else:
                time_b,time_l,integral,ampl = utility.IntegrateFullWindow(wf.copy(),baselinesRMS,ChList,i,params)
            event =[]
            channel = []

            #Write the event and channel column
            for j in range(len(integral)):
                event.append(nev)
                channel.append(ChList[i])

            #Fill the dictionaries
            dic_time_begin[ChList[i]]=time_b
            dic_time_length[ChList[i]]=time_l

            #Create auxiliary DataFrame to append it to the main dataframe
            aux = pd.DataFrame([])
            aux['event'] = event
            aux['channel'] = channel
            aux['time'] = time_b
            aux['time_len'] = time_l
            aux['integral'] = integral
            aux['ampl'] = ampl
            aux['npeaks']= npeaks
            aux['is_sat']= is_sat

            #Append the event dataframe to the main one
            #df_mast = df_mast.append(aux,ignore_index = True)
            df_mast = pd.concat([df_mast,aux],ignore_index=True)
    print(df_mast.head(100))

    #Save DF
    df_mast.to_csv(args.outfile)
    
