
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
import os
import numpy as np
from filters import filter_eeg,filter_signal



def read_data(path_info:tuple,time_point:str,dir_path:str)->(tuple,str,'(ecg_arr,abp_arr,eeg_arr),label'):
    label = path_info.label
    ecg_filename = path_info.ecg.replace('csv','npz')
    abp_filename = path_info.abp.replace('csv','npz')
    eeg_filename = path_info.eeg.replace('csv','npz')
    
    if label == 'e' :
        ecg_arr = np.load(os.path.join(dir_path,ecg_filename))['arr_0']
        abp_arr = np.load(os.path.join(dir_path,abp_filename))['arr_0']
        eeg_arr = np.load(os.path.join(dir_path,eeg_filename))['arr_0']
        label=1
    else:
        ecg_arr = np.load(os.path.join(dir_path,ecg_filename))['arr_0']
        abp_arr = np.load(os.path.join(dir_path,abp_filename))['arr_0']
        eeg_arr = np.load(os.path.join(dir_path,eeg_filename))['arr_0']
        label=0
    
    ecg_arr = filter_signal(ecg_arr,ftype='FIR',band='bandpass',order=int(0.3*500),frequency=(1,40),sampling_rate=500).copy()
    ecg_arr = (ecg_arr - ecg_arr.mean())/(ecg_arr.std()+1e-100)
    
    eeg_arr =  filter_eeg(ecg_arr,sampling_rate=128).copy()
    
    return (ecg_arr,abp_arr,eeg_arr),label 


class biosignal_dataset(Dataset):
    def __init__(self,dir_path:'root path for total data',fileinfo:'train,test info df',time_point):
        self.fileinfo = fileinfo.reset_index(drop=True)
        self.dir_path = dir_path
        self.time_point = time_point
        self.before_tensor = None
        self.before_label = None

    def __len__(self):
        return len(self.fileinfo)
    
    def __getitem__(self,idx):
        pathinfo = self.fileinfo.loc[idx]
        (ecg_arr,abp_arr,eeg_arr),label = read_data(pathinfo,self.time_point,self.dir_path)

        input_tensor = torch.from_numpy(np.r_[ecg_arr,abp_arr,eeg_arr]).float()
        label_tensor = torch.Tensor([label])
        
        if input_tensor.shape[0]<67680:
            print('Data ERR: ',pathinfo)
            input_tensor = self.before_tensor
            label_tensor = self.before_label
        
        self.before_tensor = input_tensor
        self.before_label = label_tensor

        return (input_tensor,label_tensor)



def prepare_dataset(time_point,batch_size,sqi):

    if os.name =='posix': #ubuntu
        list_dir = '~/jjong/workplace/datathon_2019/data/train_test_split/version2/'
        data_dir = '/mnt/storage2/datathon2019_3/npz_wave/'
    elif os.name =='nt': #window
        list_dir ='E:\\Jupyter_notebook\\JJH\dataStorage\\HDI\\train_test_split'
        data_dir = 'E:\\Jupyter_notebook\\JJH\\dataStorage\\HDI\\npz_wave'
    else:
        print('check os name')
        raise OSError
    print(list_dir)
    print(data_dir)

    train_df = pd.read_csv(os.path.join(list_dir,'{}M_abp_{}_ecg_{}_eeg_{}_train.csv'.format(time_point,*sqi)),index_col=0)#.sample(n=500, random_state=1)
    valid_df = pd.read_csv(os.path.join(list_dir,'{}M_abp_{}_ecg_{}_eeg_{}_val.csv'.format(time_point,*sqi)),index_col=0)#.sample(n=500, random_state=1)
    test_df  = pd.read_csv(os.path.join(list_dir,'{}M_abp_{}_ecg_{}_eeg_{}_test.csv'.format(time_point,*sqi)),index_col=0)#.sample(n=500, random_state=1)
    #15M_abp_0.30_ecg_0.30_eeg_0.30_train.csv
    #train_df, valid_df,_, _  = train_test_split(train_df,train_df.label,test_size=0.2)
 
    train_df_e = train_df.query("label=='e'")
    train_df_n = train_df.query("label=='n'")
    unbalance_rate = int(len(train_df_n)/len(train_df_e))
    train_df = pd.concat([train_df]+[train_df_e]*unbalance_rate)


    tr_dataset = biosignal_dataset(data_dir,train_df,time_point)
    vd_dataset = biosignal_dataset(data_dir,valid_df,time_point)
    te_dataset = biosignal_dataset(data_dir,test_df,time_point)

    tr_dataloader = DataLoader(tr_dataset,batch_size=batch_size,shuffle=True,num_workers=1,drop_last=True)
    vd_dataloader = DataLoader(vd_dataset,batch_size=batch_size*2,shuffle=False,num_workers=1,drop_last=False)
    te_dataloader = DataLoader(te_dataset,batch_size=batch_size*2,shuffle=False,num_workers=1,drop_last=False)

    return tr_dataloader,vd_dataloader,te_dataloader