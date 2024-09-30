import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
import numpy as np

folders = ["sp_calib_8.5bar_ch1","sp_calib_8.5bar_ch2","sp_calib_8.5bar_ch3","sp_calib_8.5bar_ch4"]
channels = ["CH1","CH2","CH3","CH4"]

df_list_temp = []

final_dfs = []

counter = 0

for folder in folders:
    filelist = os.listdir("tempdata/"+folder)

    for a_file in filelist:
        df = pd.read_hdf("tempdata/"+folder+"/"+a_file)
        df['event'] = counter
        df_list_temp.append(df)
        counter+=1
        
    final_dfs.append(pd.concat(df_list_temp, ignore_index=True))
    counter =0

    df_list_temp=[]

print(final_dfs[2])
    
# Create a figure and set of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Flatten the 2x2 array of axes for easy iteration
axs = axs.flatten()

# Iterate over each DataFrame and corresponding axis
for i, (df, ch) in enumerate(zip(final_dfs, channels)):
    # Create the 2D histogram
    h = axs[i].hist2d(df['TIME'], df[ch], bins=[250, 250], range=[[-2e-7,6e-7],[-0.2,0.5]], cmap='viridis',norm=LogNorm())
    
    # Add colorbar to each subplot
    #plt.colorbar(h[3], ax=axs[i])
    
    # Set titles and labels
    axs[i].set_title(f'{ch} vs TIME')
    axs[i].set_xlabel('TIME')
    axs[i].set_ylabel(ch)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
#plt.show()

thr = [0.12,0.15,0.2,0.16]

def filter_events(df, channel, threshold):
    time_threshold = 1.8e-7
    low_time_threshold = -0.8e-7
    waveform_limit = 0.15
    
    # Filter the DataFrame for times greater than the time_threshold
    df_time_filtered = df[(df['TIME'] > time_threshold) | (df['TIME'] < low_time_threshold)]
    
    # Find events where the waveform exceeds the waveform_limit for TIME > time_threshold
    invalid_events = df_time_filtered[df_time_filtered[channel] > waveform_limit]['event'].unique()
    
    # Calculate max values for each event in the original DataFrame
    max_values = df.groupby('event')[channel].max()
    
    # Find valid events that have max values below the threshold and are not in invalid_events
    valid_events = max_values[(max_values < threshold) & (~max_values.index.isin(invalid_events))].index
    
    # Return the filtered DataFrame containing only valid events
    return df[df['event'].isin(valid_events)]


# Example for applying to multiple DataFrames
filtered_dfs = [filter_events(df, ch, thr) for df,ch,thr in zip(final_dfs,channels,thr)]

# Create a figure and set of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Flatten the 2x2 array of axes for easy iteration
axs = axs.flatten()

# Iterate over each DataFrame and corresponding axis
for i, (df, ch) in enumerate(zip(filtered_dfs, channels)):
    # Create the 2D histogram
    h = axs[i].hist2d(df['TIME'], df[ch], bins=[250, 250], range=[[-2e-7,6e-7],[-0.2,0.5]], cmap='viridis',norm=LogNorm())
    
    # Add colorbar to each subplot
    #plt.colorbar(h[3], ax=axs[i])
    
    # Set titles and labels
    axs[i].set_title(f'{ch} vs TIME')
    axs[i].set_xlabel('TIME')
    axs[i].set_ylabel(ch)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
#plt.show()


averageval = []
timeval = []

for j, df in enumerate(filtered_dfs):
    list_val=[]
    time_val=[]
    # Determine the start and end of the original TIME range
    start_time = df['TIME'].min()
    end_time = df['TIME'].max()

    # Generate a uniformly sampled TIME vector
    time = np.linspace(start_time, end_time, round(len(df['TIME'].unique())/32))#Check

    for i in range(len(time)-1):
        df_time = df[ (df['TIME']>time[i]) & (df['TIME']<time[i+1]) ]
        list_val.append(df_time[channels[j]].mean())
        time_val.append((time[i]+time[i+1])/2)
    averageval.append(list_val)
    timeval.append(time_val)    
    
    
# Create a figure and set of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Flatten the 2x2 array of axes for easy iteration
axs = axs.flatten()

# Titles for each subplot
titles = ['Average Value 0 vs TIME', 'Average Value 1 vs TIME', 'Average Value 2 vs TIME', 'Average Value 3 vs TIME']


# Iterate over each averageval vector and corresponding axis
for i in range(4):
    #axs[i].scatter(filtered_dfs[i]['TIME'].unique(), averageval[i], s=10, c='blue', alpha=0.7)
    axs[i].scatter(timeval[i], averageval[i], s=10, c='blue', alpha=0.7)
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('TIME')
    axs[i].set_ylabel(f'averageval[{i}]')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()

for i in range(4):
    # Create a DataFrame with TIME and the corresponding averageval[i]
    df = pd.DataFrame({'TIME': timeval[i], f'averageval': averageval[i]})
    df.sort_values(by=['TIME'])
    df['averageval'] -= df[ (df['TIME']>-2e-7) & (df['TIME']<-1e-7) ]['averageval'].mean() 
    # Export to CSV
    df.to_csv(f'singlePhes/averageval_{i}_vs_TIME.csv', index=False)


