import os
def config_list():
    save_path = '/home/jjong/jjong/workplace/datathon_2019/save' if os.name=='posix'\
        else 'E:\Jupyter_notebook\JJH\workplace\Datathon_2019\save'
    device = 'cuda:0'
    time_point = 3
    channel_n = 5
    lr = 0.0005

    ecg_window = 15
    abp_window = 15
    eeg_window = 7
    sqi=['0.80','0.30','0.50'] #abp ecg eeg
    
    return [
    #'''---------------------3M------------------'''
    {
    'time_point':3,
    'device':device,
    'waves':['abp'],
    'windows':[abp_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':3,
    'device':device,
    'waves':['abp','eeg'],
    'windows':[abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':3,
    'device':device,
    'waves':['abp','ecg'],
    'windows':[abp_window,ecg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':3,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,

    #'''---------------------5M------------------'''
    {
    'time_point':5,
    'device':device,
    'waves':['abp'],
    'windows':[abp_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,    
        {
    'time_point':5,
    'device':device,
    'waves':['abp','eeg'],
    'windows':[abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':5,
    'device':device,
    'waves':['abp','ecg'],
    'windows':[abp_window,ecg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':5,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
      #'''-------------------10M--------------------'''
    {
    'time_point':10,
    'device':device,
    'waves':['abp'],
    'windows':[abp_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,    
        {
    'time_point':10,
    'device':device,
    'waves':['abp','eeg'],
    'windows':[abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':10,
    'device':device,
    'waves':['abp','ecg'],
    'windows':[abp_window,ecg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':10,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,

       #'''---------------------15M------------------'''
    {
    'time_point':15,
    'device':device,
    'waves':['abp'],
    'windows':[abp_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,    
        {
    'time_point':15,
    'device':device,
    'waves':['abp','eeg'],
    'windows':[abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':15,
    'device':device,
    'waves':['abp','ecg'],
    'windows':[abp_window,ecg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':15,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,

       #'''---------------------20M------------------'''
    {
    'time_point':20,
    'device':device,
    'waves':['abp'],
    'windows':[abp_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,    
        {
    'time_point':20,
    'device':device,
    'waves':['abp','eeg'],
    'windows':[abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':20,
    'device':device,
    'waves':['abp','ecg'],
    'windows':[abp_window,ecg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,
        {
    'time_point':20,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi,
    'save_path':save_path
    }
    ,

    ]