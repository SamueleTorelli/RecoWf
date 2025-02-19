
import pandas as pd
import re

def parse_txt_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    event_num = None
    event_time = None
    time_step = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Capture Event number
        if line.startswith("Event n."):
            event_num = int(re.search(r'\d+', line).group())
        
        # Capture TimeStamp
        elif line.startswith("TimeStamp:"):
            event_time = int(re.search(r'\d+', line).group())
        
        # Capture 1 Sample step value
        elif line.startswith("1 Sample ="):
            time_step = float(re.search(r'0\.[0-9]+', line).group())
            
        # Capture data values
        elif re.match(r'^\d+\s+-?\d+\.\d+', line):
            s, ch2 = line.split()
            s = int(s)
            ch2 = float(ch2)/1000
            time = s * time_step/1e6
            data.append((time, ch2, event_num, event_time))
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["TIME", "CH2", "event", "event_time"])
    return df


def parse_txt_to_dataframe_multich(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    event_num = None
    event_time = None
    time_step = None
    channels = []
    
    for i, line in enumerate(lines):
        line = line.strip()

        # Capture Event number
        if line.startswith("Event n."):
            event_num = int(re.search(r'\d+', line).group())
                        
        # Capture TimeStamp
        elif line.startswith("TimeStamp:"):
            event_time = int(re.search(r'\d+', line).group())

        # Capture 1 Sample step value
        elif line.startswith("1 Sample ="):
            time_step = float(re.search(r'0\.[0-9]+', line).group())

        # Capture channels
        elif line.startswith("S") and "CH:" in line :
            channels = [f"CH{i+1}" for i in range(line.count("CH:"))]

        # Capture data values
        elif re.match(r'^\d+', line):
            values = line.split()
            s = int(values[0])
            time = s * time_step / 1e6
            ch_values = [float(v) / 1000 for v in values[1:]]
            data.append((time, *ch_values, event_num, event_time))

    # Create DataFrame
    column_names = ["TIME"] + channels + ["event", "event_time"]
    df = pd.DataFrame(data, columns=column_names)
    return df
