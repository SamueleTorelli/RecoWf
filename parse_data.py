
import pandas as pd
import re
import struct
from pathlib import Path
import numpy as np

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


def load_waveforms_until_eof(
    path,
    *,
    channels=4,
    samples_per_waveform=500,
    dtype="<f4",          # little-endian float32
    event_header_bytes=28 # set to 0 if there's no per-event header
):
    """
    Reads events until EOF.
    Each event layout: [event_header_bytes] + [channels * samples_per_waveform * dtype]
    Returns:
      waveforms: (num_events, channels, samples_per_waveform) array
      headers:   (num_events, event_header_bytes//4) <u4 array (or None if header_bytes==0)
    """
    path = Path(path)
    sample_bytes = np.dtype(dtype).itemsize
    data_bytes_per_event = channels * samples_per_waveform * sample_bytes

    wfs = []
    hdrs = [] if event_header_bytes else None
    event_times = [] if hdrs is not None else None

    with path.open("rb") as f:
        evt = 0
        while True:
            # Read per-event header (if any)
            if event_header_bytes:
                h = f.read(event_header_bytes)
                if not h:
                    break  # clean EOF at boundary
                if len(h) != event_header_bytes:
                    print(f"Warning: partial header at event {evt} — stopping.")
                    break
                if event_header_bytes >= 12:
                    event_id = struct.unpack("<I", h[0:4])[0]
                    event_time = struct.unpack("<Q", h[4:12])[0]
            else:
                # If no header, peek one byte to see if we're at EOF
                p = f.peek(1) if hasattr(f, "peek") else f.read(1)
                if p == b"":
                    break
                if not hasattr(f, "peek"):
                    # we consumed 1 byte; seek back
                    f.seek(-1, 1)

            # Read waveform payload
            buf = f.read(data_bytes_per_event)
            if len(buf) != data_bytes_per_event:
                print(f"Warning: partial data payload at event {evt} — stopping.")
                break

            arr = np.frombuffer(buf, dtype=dtype).reshape(channels, samples_per_waveform)
            wfs.append(arr)
            event_times.append(event_time) if hdrs is not None else None
            evt += 1

    if not wfs:
        raise RuntimeError("No complete events found.")

    waveforms = np.stack(wfs, axis=0)  # (E, C, N)
    # headers = (np.stack(hdrs, axis=0) if hdrs is not None else None)
    return waveforms, event_times


def waveforms_to_dataframe(arr):
    sample_width_ns = 6.25

    arr = np.asarray(arr)

    if arr.ndim != 3:
        raise ValueError(
            f"Expected a 3D array (event, channel, sample), got shape {arr.shape}"
        )

    n_events, n_channels, n_samples = arr.shape

    # Event index
    events = np.repeat(np.arange(n_events), n_samples)

    # Time axis (microseconds), unchanged
    sample_nums = np.tile(np.arange(n_samples), n_events)
    time = sample_nums * sample_width_ns / 1000.0

    # (event, channel, sample) -> (event, sample, channel)
    data = arr.transpose(0, 2, 1).reshape(-1, n_channels)

    df = pd.DataFrame(
        data,
        columns=[f"CH{ch}" for ch in range(n_channels)]
    )

    # Multiply selected channels by -1
    for ch in [3, 5, 7, 9]:
        col = f"CH{ch}"
        if col in df.columns:
            df[col] *= -1

    # Column order: TIME | CH* | event
    df.insert(0, "TIME", time)
    df.insert(len(df.columns), "event", events)

    return df