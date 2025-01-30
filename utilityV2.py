import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq, ifft

pheshape = { 'CH1' : pd.read_csv("singlePhes/averageval_0_vs_TIME.csv"),
    'CH2' : pd.read_csv("singlePhes/averageval_1_vs_TIME.csv"),
    'CH3' : pd.read_csv("singlePhes/averageval_2_vs_TIME.csv"),
    'CH4' : pd.read_csv("singlePhes/averageval_3_vs_TIME.csv")
}

def remove_comments(file_content):
    cleaned_lines = []
    for line in file_content:
        # Remove anything after '#'
        cleaned_line = line.split('#')[0].strip()
        if cleaned_line:  # Only add non-empty lines
            cleaned_lines.append(cleaned_line)
    return '\n'.join(cleaned_lines)


def replace_inf_with_max(wf, columns):

    for column in columns:
        # Check if the column contains 'inf'
        if np.any(np.isinf(wf[column])):
            # Find the maximum value in the column that is less than 'inf'
            max_value = wf[column][wf[column] < np.inf].max()
            min_value = wf[column][wf[column] > -np.inf].min()
            # Replace 'inf' with the maximum value
            wf[column] = wf[column].replace([np.inf], max_value)
            
            # Replace '-inf' with the minimum value
            wf[column] = wf[column].replace([-np.inf], min_value)


            
def PlotWfs(wf,sfbegin,params):
    # Create a figure with four subplots
    fig, axs = plt.subplots(len(wf.columns[1:-1].tolist()), 1, figsize=(10, 5), sharex=True)
    
    if(len(wf.columns[1:-1].tolist())==1):
        ch=wf.columns[1:-1].tolist()
        axs.scatter(wf["TIME"], wf[ch[0]], label=ch[0], marker='o', s=1, edgecolors="red")
        axs.axvline(x=wf["TIME"].iloc[sfbegin[0]],linewidth=1, color='b')
        axs.axvline(x=wf["TIME"].iloc[sfbegin[0]+params["n_points_pre_wf"]],linewidth=1, color='r')
        axs.set_ylabel('V [V]')
        axs.legend()
        
    else:
        for i, ch in enumerate(wf.columns[1:-1].tolist()):    
            # Scatter plot for Channel 1
            #print(wf["TIME"].iloc[sfbegin[i]])
            axs[i].scatter(wf["TIME"], wf[ch], label=ch, marker='o', s=1, edgecolors="red")
            axs[i].axvline(x=wf["TIME"].iloc[sfbegin[i]],linewidth=1, color='b')
            axs[i].axvline(x=wf["TIME"].iloc[sfbegin[i]+params["n_points_pre_wf"]],linewidth=1, color='r')
            axs[i].set_ylabel('V [V]')
            axs[i].legend()
        
    # Set the title for the entire figure
    fig.suptitle('Voltage vs time', y=0.92)
    
    # Show the plot
    plt.show()

def plot_scaled_phe( chlist, chindex, A_mid, b, e, sat_thr):
    # Extract the relevant DataFrame
    df = pheshape[chlist[chindex]]
    
    # Calculate the scaled values
    scaled_values = df['averageval'] * A_mid
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot averageval * A_mid vs TIME
    plt.plot(df['TIME'], scaled_values, label=f"{chlist[chindex]}: averageval * {A_mid:.2f}", color='blue',
             linestyle='None', marker='.')
    
    # Plot vertical lines at x = b and x = e
    plt.axvline(x=b, color='red', linestyle='--', label=f'x = {b}')
    plt.axvline(x=e, color='red', linestyle='--', label=f'x = {e}')
    
    # Plot horizontal line at y = sat_thr[chlist[chindex]]
    plt.axhline(y=sat_thr, color='green', linestyle='--', label=f'y = {sat_thr}')
    
    # Add labels and title
    plt.xlabel('TIME')
    plt.ylabel('averageval * A_mid')
    plt.title(f"Plot of {chlist[chindex]} with Scaling Factor A_mid = {A_mid:.2f}")
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.grid(True)
    
def BuildChList(par): #DEPRECATED

    #Build the channel list with channel names
    
    chList = []
    for i in range(par["nchannels"]):
        chList.append("CH"+str(i+1));

    return chList


def is_saturated(df,b,e,chlist,chindex,par):

    sat_thr = eval(par['sat_lv'])

    sat_thr['CHSum'] = sum(sat_thr.values())
    
    df_range = df[(df['TIME']>b) & (df['TIME']<e)]

    sat_length_pt = (df_range[chlist[chindex]] > sat_thr[chlist[chindex]]).sum()
    
    if( sat_length_pt  >35  ):
        time_start = df_range[ (df_range[chlist[chindex]] > sat_thr[chlist[chindex]]) ]['TIME'].iloc[0]
        time_end = df_range[ (df_range[chlist[chindex]] > sat_thr[chlist[chindex]]) ]['TIME'].iloc[-1]
        sat_length = time_end-time_start
        return sat_length, time_start, time_end 
    else:
        return 0, 0, 0

def saturated_integral(sat_length,b,e,chlist,chindex,params):

    sat_thr = eval(params['sat_lv'])
    
    A_min = 15
    A_max = 200

    iter_count = 0

    while (A_max - A_min) > 0.01 and iter_count < 30:
        iter_count += 1
        A_mid = (A_min + A_max) / 2
        
        tempdf=pheshape[chlist[chindex]].copy()
        #print(A_mid)
        tempdf['averageval']*=A_mid
        
        tempdf=tempdf[ tempdf['averageval'] > sat_thr[chlist[chindex]]  ]

        try:
            time_above_threshold = tempdf['TIME'].iloc[-1]-tempdf['TIME'].iloc[0]
        except:
            time_above_threshold=0

        if abs(time_above_threshold - sat_length)<1e-10:
            break  # Found exact match
        elif (time_above_threshold - sat_length)>0:
            A_max = A_mid  # A_mid is too large
        else:
            A_min = A_mid  # A_mid is too small

    bin_width = pheshape[chlist[chindex]]['TIME'].iloc[1] - pheshape[chlist[chindex]]['TIME'].iloc[0]
    corr_integral = ((tempdf['averageval']*A_mid).sum() - len(tempdf['averageval'])*sat_thr[chlist[chindex]])*bin_width

    plot_scaled_phe(chlist, chindex, A_mid, tempdf['TIME'].iloc[0], tempdf['TIME'].iloc[-1], sat_thr[chlist[chindex]])
    
    return corr_integral


    
def Analyze(df,rms,chlist,chindex,params):

    time_start = df['TIME'].iloc[0]
    time_end = df['TIME'].iloc[-1]
    
    bin_width = (df['TIME'].iloc[1]-df['TIME'].iloc[0])
    
    # Reduce the search of the waveforms
    #df = df[(df['TIME'] > time_start) & (df['TIME'] < time_end)]
    
    # Full DataFrame with the boolean condition
    condition = df[chlist[chindex]] > params['nsigma'] * rms[chindex]
    
    transitions = condition.ne(condition.shift()).cumsum()

    # Extract times for transitions
    t_begin = df['TIME'][condition & (transitions.diff() == 1)].tolist()
    t_end = df['TIME'][~condition & (transitions.diff() == 1)].tolist()

    if(len(t_end)>0 and len(t_begin)>0):
        if(t_end[0]<t_begin[0]):
            #t_end.pop(0)
            t_begin.append(time_start)
    if(len(t_end)<len(t_begin)):
        t_end.append(time_end)

    t_length =[]

    #print(t_begin)
    #Calculate the lenght of the integration window
    if t_begin and t_end:
        t_begin.sort(),t_end.sort()

    for b,e in zip(t_begin,t_end):
        t_length.append(e-b)
    
    integral = []
    amplitude = []
    npeaks = []
    is_sat = []
    
    #Compute the integral
    for b,e in zip(t_begin, t_end):
        sat_len, sat_b, sat_e = is_saturated(df,b,e,chlist,chindex,params) 

        if(sat_len):
            is_sat.append(True)

            if(params['sat_corr']):
                corr_int = saturated_integral(sat_len,sat_b,sat_e,chlist,chindex,params)
            else:
                corr_int=0
        else:
            is_sat.append(False)
            corr_int = 0
                      
        df_A=df[df['TIME']>b]
        #print(f"int = {(df_A[df_A['TIME']<e][chlist[chindex]].sum())*bin_width}, corr={corr_int}")
        integral.append( (df_A[df_A['TIME']<e][chlist[chindex]].sum())*bin_width+corr_int)
        #print(b,e,integral)
        
        amplitude.append( df_A[df_A['TIME']<e][chlist[chindex]].max())
        npeaks.append(len(find_peaks(
            df_A[df_A['TIME']<e][chlist[chindex]],
            width=params['peaks_width'],height=params['peaks_height'])[0]))

    #print(chlist[chindex],is_sat)
        
    return t_begin,t_length,integral,amplitude,npeaks,is_sat


def IntegrateFullWindow(df,rms,chlist,chindex,params):

    search_ranges = eval(params['search_ranges'])
    
    # Reduce the search of the waveforms
    df = df[(df['TIME'] > search_ranges[chlist[chindex]][0]) & (df['TIME'] < search_ranges[chlist[chindex]][1])]
    
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
    fig, axs = plt.subplots(len(wf.columns[1:-1].tolist()), 1, figsize=(7, 5), sharex=True)

    if(len(wf.columns[1:-1].tolist())==1):
        ch=wf.columns[1:-1].tolist()
        
        axs.scatter(wf["TIME"], wf[ch[0]], label=ch[0], marker='.', s=1, edgecolors='black')
        for j in range(len(dic[ch[0]])):
            axs.axvline(x=dic[ch[0]][j],linewidth=1, color='b')
            axs.axvline(x=dic[ch[0]][j]+dic_len[ch[0]][j],linewidth=.3, color='r') 
        axs.axhline(y=par["nsigma"]*rms[0])    
        axs.set_ylabel('V [V]')
        axs.legend()
    else:
        for i, ch in enumerate(wf.columns[1:-1].tolist()):
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
    for i in range(len(wf.columns[1:-1].tolist())):
        t_rms.append([-1,9999])
    
    for i in range(0,len(wf)-params["n_points_pre_wf"],50):
        rms_in_range = wf[wf.columns[1:-1]].iloc[20+i:20+i + params["n_points_pre_wf"]].std().values
        for j,rms in enumerate(rms_in_range):
            if rms < t_rms[j][1]:
                t_rms[j][0]=20+i
                t_rms[j][1]=rms

    #print(t_rms)
    
    time=[]
    rms=[]
    
    for values in t_rms:
        time.append(values[0])
        rms.append(values[1])
        
    return time,rms


def RemoveNoiseFourier(wf,freq_cut):

    sampling_interval = wf['TIME'].iloc[1] - wf['TIME'].iloc[0]
    sampling_rate = 1 / sampling_interval
    N = len(wf['TIME'])
    
    cols=wf.columns[1:].tolist()
    
    for i in range(len(cols)):
        signal = wf[cols[i]].values

        # Perform Fourier Transform
        yf = fft(signal)
        xf = fftfreq(N, sampling_interval)

        # Apply the low-pass filter (zero out frequencies above cutoff)
        yf[np.abs(xf) > freq_cut] = 0

        # Inverse FFT to obtain the filtered signal in time domain
        filtered_signal = ifft(yf).real

        wf[cols[i]]=filtered_signal


def CreateWfSum(wf,params,baselines,baselinesRMS):


    amp_factors = eval(params['amp_factors'])
    avg_amp_factor = np.mean(list(amp_factors.values()))  # Average amplification factor
    
    if params["is_amplified"] == False:
        # Calculate CHSum with amplification adjustments
        wf['CHSum'] = sum(  wf[ch] / amp_factors[ch] * avg_amp_factor for ch in amp_factors.keys()  )
    else:
        # Simple sum of the channels
        wf['CHSum'] = sum(wf[ch] for ch in amp_factors.keys())

    baselines.append(0)
    baselinesRMS.append(np.sqrt(sum(x**2 for x in baselinesRMS)))
    
    # Return the dataframe with the required columns
    return wf[['TIME', 'CH1', 'CH2', 'CH3', 'CH4', 'CHSum', 'event']]




