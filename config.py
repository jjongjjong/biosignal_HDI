
def config_list():
    
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
    # {
    # 'time_point':3,
    # 'device':device,
    # 'waves':['abp'],
    # 'windows':[abp_window],
    # 'channel_n':channel_n,
    # 'lr':lr,
    # 'sqi':sqi
    # }
    # ,    
    #     {
    # 'time_point':3,
    # 'device':device,
    # 'waves':['abp','eeg'],
    # 'windows':[abp_window,eeg_window],
    # 'channel_n':channel_n,
    # 'lr':lr,
    # 'sqi':sqi
    # }
    # ,
    #     {
    # 'time_point':3,
    # 'device':device,
    # 'waves':['abp','ecg'],
    # 'windows':[abp_window,ecg_window],
    # 'channel_n':channel_n,
    # 'lr':lr,
    # 'sqi':sqi
    # }
    # ,
        {
    'time_point':3,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
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
    'sqi':sqi
    }
    ,    
        {
    'time_point':5,
    'device':device,
    'waves':['abp','eeg'],
    'windows':[abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
    }
    ,
        {
    'time_point':5,
    'device':device,
    'waves':['abp','ecg'],
    'windows':[abp_window,ecg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
    }
    ,
        {
    'time_point':5,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
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
    'sqi':sqi
    }
    ,    
        {
    'time_point':10,
    'device':device,
    'waves':['abp','eeg'],
    'windows':[abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
    }
    ,
        {
    'time_point':10,
    'device':device,
    'waves':['abp','ecg'],
    'windows':[abp_window,ecg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
    }
    ,
        {
    'time_point':10,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
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
    'sqi':sqi
    }
    ,    
        {
    'time_point':15,
    'device':device,
    'waves':['abp','eeg'],
    'windows':[abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
    }
    ,
        {
    'time_point':15,
    'device':device,
    'waves':['abp','ecg'],
    'windows':[abp_window,ecg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
    }
    ,
        {
    'time_point':15,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
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
    'sqi':sqi
    }
    ,    
        {
    'time_point':20,
    'device':device,
    'waves':['abp','eeg'],
    'windows':[abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
    }
    ,
        {
    'time_point':20,
    'device':device,
    'waves':['abp','ecg'],
    'windows':[abp_window,ecg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
    }
    ,
        {
    'time_point':20,
    'device':device,
    'waves':['ecg','abp','eeg'],
    'windows':[ecg_window,abp_window,eeg_window],
    'channel_n':channel_n,
    'lr':lr,
    'sqi':sqi
    }
    ,

    ]