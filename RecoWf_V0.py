import pandas as pd
import matplotlib.pyplot as plt
import json
import utility
import baseline_cal
import numpy as np
from scipy.ndimage import gaussian_filter1d    
import os
import argparse
import sys
    
#Load reco parameters
if __name__== '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--inputdir", help="Input dir containing the csv files")
    parser.add_argument("-O", "--outfile", help="Output file name")
    parser.add_argument("-C", "--config", help="Configuration file")
    args = parser.parse_args()

    #Read analysis parameters from file
    with open(args.config) as f:
        params = json.load(f)

    
    if not args.inputdir or not args.outfile:
        print("python3.8 RwcoWF_V0.py -i /path/to/input/folder -o outfilename")
        sys.exit(1)
    
    #Read files in the folder
    files = os.listdir(args.inputdir)

    #Initialize output dataframe
    df_mast = pd.DataFrame(columns=['event','channel','time','time_len','integral'])

    baselines = []
    baselinesRMS=[]

    if(params["variable_mean_rms"]==False):
        baselines,baselinesRMS = baseline_cal.mean_baselane(args.inputdir,params)
    
    #Loop all over the files
    for nev,a_file in enumerate(files):
        if nev%10==0:
            print(nev)
        
        #Open df
        wf = pd.read_csv(args.inputdir+"/"+a_file,skiprows=15)

        ChList = wf.columns[1:].tolist()
        
        #If true applies the Mean Filter to each channel
        if params['filter']==True:
            for i in range(len(ChList)):
                wf[ChList[i]]=utility.MeanFilter(wf[ChList[i]],10)

        
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

        
        dic_time_begin={}
        dic_time_length={}
    
        for ch in ChList:
            dic_time_begin[ch]=[]
            dic_time_length[ch]=[]    
        
        for i in range(len(ChList)):
            #Analyze the waveform
            if( not params['full_window']):
                time_b,time_l,integral,ampl,npeaks = utility.Analyze(wf.copy(),baselinesRMS,ChList,i,params)
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

            #Append the event dataframe to the main one
            #df_mast = df_mast.append(aux,ignore_index = True)
            df_mast = pd.concat([df_mast,aux],ignore_index=True)
    print(df_mast.head(100))

    #Save DF
    df_mast.to_csv(args.outfile)
    
