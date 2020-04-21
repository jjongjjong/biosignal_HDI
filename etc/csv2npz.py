import numpy as np
import pandas as pd
import os
import pathlib
from multiprocessing import Process

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

            
def csv2npz(path):
    tokens = path.split('/')
    folder_path = os.path.join(*tokens[:-1])
    filename = tokens[-1]
    
    save_folder = '/'+folder_path.replace('datathon2019','datathon2019_2')
    npz_name = filename.replace('.csv','')
    
    signal_series = pd.read_csv(path,index_col=0).iloc[:,0]
    if (signal_series.name =='SNUADC/ECG_II') or (signal_series.name=='SNUADC/ART'):
        signal_arr = signal_series.values[:30000]
    elif signal_series.name =='BIS/EEG1_WAV':
        signal_arr = signal_series.values[:7680]
    else:
        print('not proper signal name')
    
    pathlib.Path(save_folder).mkdir(exist_ok=True,parents=True)
    np.savez(os.path.join(save_folder,npz_name),x=signal_arr)
    
    print(path,'  Done!')
    return


def csv2npz_proc(path_list):
    for path in path_list:
        csv2npz(path)
        
        
if __name__=='__main__':
    folder_dir = '/mnt/storage2/datathon2019/'
    path_gen = absoluteFilePaths(folder_dir)

    path_list  = [path for path in path_gen]
    path_arr = np.array(path_list)

    cpu_num=12
    path_split_list = np.array_split(path_arr,12)

    procs = []

    for idx,paths in enumerate(path_split_list):
        proc = Process(target=csv2npz_proc,args=(paths,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()