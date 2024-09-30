import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def PlotWfs(wf,sfbegin,params):
    # Create a figure with four subplots
    fig, axs = plt.subplots(len(wf.columns[1:].tolist()), 1, figsize=(10, 5), sharex=True)

    if(len(wf.columns[1:].tolist())==1):

        ch=wf.columns[1:].tolist()
        axs.scatter(wf["TIME"], wf[ch[0]], label=ch[0], marker='o', s=1, edgecolors="red")
        axs.axvline(x=wf["TIME"].iloc[sfbegin[0]],linewidth=1, color='b')
        axs.axvline(x=wf["TIME"].iloc[sfbegin[0]+params["n_points_pre_wf"]],linewidth=1, color='r')
        axs.set_ylabel('V [V]')
        axs.legend()
        
    else:
        for i, ch in enumerate(wf.columns[1:].tolist()):    
            # Scatter plot for Channel 1
            axs[i].scatter(wf["TIME"], wf[ch], label=ch, marker='o', s=1, edgecolors="red")
            axs[i].axvline(x=wf["TIME"].iloc[sfbegin[i]],linewidth=1, color='b')
            axs[i].axvline(x=wf["TIME"].iloc[sfbegin[i]+params["n_points_pre_wf"]],linewidth=1, color='r')
            axs[i].set_ylabel('V [V]')
            axs[i].legend()
        
    # Set the title for the entire figure
    fig.suptitle('Voltage vs time', y=0.92)
    
    # Show the plot
    plt.show()

    
def BuildChList(par):

    #Build the channel list with channel names
    
    chList = []
    for i in range(par["nchannels"]):
        chList.append("CH"+str(i+1));

    return chList


def Analyze(df,rms,chlist,chindex,params):

    search_ranges = eval(params['search_ranges'])

    time_start = search_ranges[chlist[chindex]][0]
    time_end = search_ranges[chlist[chindex]][1]

    # Reduce the search of the waveforms
    df = df[(df['TIME'] > time_start) & (df['TIME'] < time_end)]
    
    # Full DataFrame with the boolean condition
    condition = df[chlist[chindex]] > params['nsigma'] * rms[chindex]
    
    transitions = condition.ne(condition.shift()).cumsum()

    # Extract times for transitions
    t_begin = df['TIME'][condition & (transitions.diff() == 1)].tolist()
    t_end = df['TIME'][~condition & (transitions.diff() == 1)].tolist()

    if(len(t_end)>0 and len(t_begin)>0):
        if(t_end[0]<t_begin[0]):
            t_end.pop(0)
    
    if(len(t_end)<len(t_begin)):
        t_end.append(time_end)
    t_length =[]

    #Calculate the lenght of the integration window
    for b,e in zip(t_begin,t_end):
        t_length.append(e-b)
    
    integral = []
    amplitude = []
    npeaks = []
    
    #Compute the integral
    for b,e in zip(t_begin, t_end):
        #df[ (df['TIME']>b and df['TIME']<e) ][chlist[chindex]]
        df_A=df[df['TIME']>b]
        integral.append( df_A[df_A['TIME']<e][chlist[chindex]].sum())
        amplitude.append( df_A[df_A['TIME']<e][chlist[chindex]].max())
        npeaks.append(len(find_peaks(
            df_A[df_A['TIME']<e][chlist[chindex]],
            width=params['peaks_width'],height=params['peaks_height'])[0]))
        
    return t_begin,t_length,integral,amplitude,npeaks

def IntegrateFullWindow(df,rms,chlist,chindex,params):

    # Reduce the search of the waveforms
    df = df[(df['TIME'] > params["search_range"][0]) & (df['TIME'] < params["search_range"][1])]
    
    integral = []
    amplitude = []
    t_begin = []
    t_length = []
    integral.append(df[chlist[chindex]].sum())
    amplitude.append(df[chlist[chindex]].max())
    t_begin.append(-99)
    t_length.append(-99)

    return t_begin,t_length,integral,amplitude

def PlotWfsTimestamps(wf,dic,dic_len,rms,par):
    # Create a figure with four subplots
    fig, axs = plt.subplots(len(wf.columns[1:].tolist()), 1, figsize=(7, 5), sharex=True)

    if(len(wf.columns[1:].tolist())==1):
        ch=wf.columns[1:].tolist()
        
        axs.scatter(wf["TIME"], wf[ch[0]], label=ch[0], marker='.', s=1, edgecolors='black')
        for j in range(len(dic[ch[0]])):
            axs.axvline(x=dic[ch[0]][j],linewidth=1, color='b')
            axs.axvline(x=dic[ch[0]][j]+dic_len[ch[0]][j],linewidth=.3, color='r') 
        axs.axhline(y=par["nsigma"]*rms[0])    
        axs.set_ylabel('V [V]')
        axs.legend()
    else:
        for i, ch in enumerate(wf.columns[1:].tolist()):
            # Scatter plot for Channel 1 with timestamps
            axs[i].scatter(wf["TIME"], wf[ch], label=ch, marker='.', s=1, edgecolors='black')
            for j in range(len(dic[ch])):
                axs[i].axvline(x=dic[ch][j],linewidth=1, color='b')
                axs[i].axvline(x=dic[ch][j]+dic_len[ch][j],linewidth=.3, color='r') 
            axs[i].axhline(y=par["nsigma"]*rms[i])    
            axs[i].set_ylabel('V [V]')
            axs[i].legend()
        
    # Set the title for the entire figure
    fig.suptitle('Voltage vs time', y=0.92)
    
    # Show the plot
    plt.show()


def MeanFilter(ser,kern_size):

    #Apply a mean filter to the waveform
    k=kern_size
    kern=np.ones(2*k+1)/(2*k+1)
    arr=np.random.random((10))
    ser=np.convolve(ser,kern, mode='same')
    return ser

def FindSignalFreeRegion(wf,params):

    t_rms = []
    for i in range(len(wf.columns[1:].tolist())):
        t_rms.append([-1,9999])
    
    for i in range(0,len(wf)-params["n_points_pre_wf"],50):
        rms_in_range = wf[wf.columns[1:]].iloc[20+i:20+i + params["n_points_pre_wf"]].std().values
        for j,rms in enumerate(rms_in_range):
            if rms < t_rms[j][1]:
                t_rms[j][0]=20+i
                t_rms[j][1]=rms

    print(t_rms)
    
    time=[]
    rms=[]
    
    for values in t_rms:
        time.append(values[0])
        rms.append(values[1])
        
    return time,rms


