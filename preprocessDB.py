import pandas as pd
import re
import numpy as np

def PreprocessDB(filename, CH_list):

    pattern = [r"[0-9]+",r"[0-9]+",r"[0-9]+",r"0\.[0-9]+",False]
    returntype = [lambda x: int(x),lambda x: int(x),lambda x: int(x),lambda x: float(x)]

    param_list = []

    df_out = pd.DataFrame(columns=['TIME']+CH_list+["event","event_time"])
    
    line_num = 0
    with open(filename, "r") as file:
        while (line := file.readline()):
                    
            if(pattern[line_num%5]):
                match = re.search(pattern[line_num%5], line)
                param_list.append(returntype[line_num%5](match.group()))
            else:
                values =  [float(v) for v in line.split()]

                dftemp = pd.DataFrame(columns=['TIME']+CH_list+["event", "event_time"])
                                                
                dftemp[CH_list[0]] = values
                dftemp['TIME'] = np.arange(0, param_list[2]*param_list[3]*1e-6, param_list[3]*1e-6)
                # Set the 'event' and 'event_time' columns using broadcasting
                dftemp['event'] = param_list[0]
                dftemp['event_time'] = param_list[1]

                df_out = pd.concat([df_out, dftemp], ignore_index=True)
                                
                param_list.clear()
            line_num+=1

    return dftemp
            
            
