
import pandas as pd
import re
import struct

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



def parse_wf_from_binary(filename):
    data_list = []
    nlines=0
    nevents=2000
    with open(filename, "rb") as f:
        while True:
            # Read the header
            data = f.read(4)  # Read uint32_t EVID
            if not data:
                break
            EVID = struct.unpack("<I", data)[0]
            data = f.read(8)  # Read uint64_t T
            if not data:
                break
            T = struct.unpack("<Q", data)[0]
            data = f.read(4)  # Read uint32_t size
            if not data:
                break
            size = struct.unpack("<I", data)[0]
            data = f.read(8)  # Read uint64_t sampl_time
            if not data:
                break
            sampl_time = struct.unpack("<Q", data)[0]
            data = f.read(4)  # Read uint32_t ch (number of channels)
            if not data:
                break
            ch = struct.unpack("<I", data)[0]
            waveform_data = {}
            # Read waveforms for each channel
            for channel in range(ch):
                data = f.read(2)  # Read uint16_t numch
                if not data:
                    break
                numch = struct.unpack("<H", data)[0]
                channel_waveforms = []
                for _ in range(size):
                    data = f.read(4)  # Read float w
                    if not data:
                        break
                    w = struct.unpack("<f", data)[0]
                    channel_waveforms.append(w)
                waveform_data[f'{numch}'] = channel_waveforms
            # Create a row per sample point with all channels aligned
            for i in range(size):
                row = {}
                
                row.update({f'CH{j+1}': waveform_data[f'{numch}'][i]/1e3 for j,numch in enumerate(waveform_data)})
                row.update({"event": EVID})
                row.update({"event_time": T})

                data_list.append(row)

    print(nlines,nevents)
    df = pd.DataFrame(data_list)
    df.insert(0, 'TIME', (df.index % size + 1) * sampl_time/1e9)  # Time in microseconds
    return df
