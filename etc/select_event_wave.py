import sys
import pandas as pd
import matplotlib.pyplot as plt
import csv
from dateutil.parser import parse 
import io
import os
import numpy as np
from multiprocessing import Process
pd.set_option('precision',20)

from os import listdir
from os.path import isfile, join


THRESHOLD_E = 65 ## 이벤트 기준 MBP <65 mmhg 미만
THRESHOLD_N = 75 
TIME_WINDOW = 30 ## unit: minute
NONEVENT_STEP = 2 ## unit: minute
TIME_OFFSET = parse("2100-01-01 00:00:00.000").timestamp() ## 기준 time stamp
FILE_ROOT_DIR = '/mnt/storage1/data/VITALDB/' ## vital sign 경로
SAVE_ROOT_DIR = '/mnt/biosignal_DB/vital_db/extracted/'
#FILE_ROOT_DIR = '/Volumes/Fast SSD/sign/'
#SAVE_ROOT_DIR = '/Volumes/Fast SSD/event_wave/'
VITAL_TRKS = pd.read_csv('/home/jjong/jjong/workplace/datathon_2019/data/VITAL_TRKS.csv') ## vital track 경로
VITAL_TRKS = VITAL_TRKS[VITAL_TRKS['gain']>0]
CASE_IDS = []
with open ('/home/jjong/jjong/workplace/datathon_2019/data/case_id_mbp.list','r',encoding='utf-8') as f:
    CASE_IDS.extend(f.read().split())
#CASE_IDS = [int(f[:5]) for f in listdir(FILE_ROOT_DIR) if (isfile(join(FILE_ROOT_DIR, f))) & ('Solar8000_ART_MBP' in f)]
THREADS = 12
CHK_POINT = [0,3,5,10,15,20] ## unit: minute


def surround_mbps (evt_idx, event_surround):
    e_sec = []
    e_mbp = []
    for e in event_surround:
        if e['event_idx']==evt_idx:
            e_sec.extend(e['second'])
            e_mbp.extend(e['mbps'])
    return e_sec, e_mbp
def event_selection(c_id, id_trks):
    time_table = dict() ## 필요한 time에 mbp만 가져오기
    event_cand = list()
    event_cand_surround = list()
    nonevents = list()
    mbps_window = list()
    
    npz_path = FILE_ROOT_DIR+'{:05d}'.format(c_id)+'_Solar8000_ART_MBP.npz' # A-line file 읽기
    try:
        value = np.load(npz_path)['arr_0'] # sign
    except: 
        return event_cand_surround, event_cand, nonevents 
    
    mbp_dtstart = id_trks['dtstart'][(id_trks['tname']=='Solar8000/ART_MBP')] # A-line 측정 시간
    df_mbp = pd.DataFrame(value) ## pandas로 만들기 
    df_mbp = df_mbp.rename({0:'timestamp',1:'ART_MBP'},axis=1) ## column 이름 변경
    #df_mbp = df_mbp.sort_values('timestamp',ascending=True)
    try:
        df_mbp["second"] = df_mbp['timestamp'].apply(lambda x: x+mbp_dtstart - TIME_OFFSET) # 2100년 1월1일 0시기준으로 time stamp 
    except: 
        print('two more cases: ',c_id,'\n',mbp_dtstart)
        return event_cand_surround, event_cand, nonevents
        
    df_mbp["minute"] = df_mbp["second"].apply(lambda x: int(x / 60)) ## 초 --> 분

    # 1. 이상치 MBP 값 제거  (5이하/ 160이상 제거)
    df_mbp = df_mbp[(df_mbp['ART_MBP']>=5) &(df_mbp['ART_MBP']<=160)]

    # 2. 분당 25회 미만 카운트 제거 (기준: 25회)
    cnt_mbp_min = df_mbp[['minute','ART_MBP']].groupby(by=['minute']).count().reset_index() # 분별 개수 체크
    skip_min_list = list(cnt_mbp_min['minute'][(cnt_mbp_min['ART_MBP']<25)]) # 스킵해야할 minute 리스트
    
    # 3-1. 1분 단위 MBP 저장
    for i in range(len(df_mbp)): 
        key = df_mbp["minute"].iloc[i] 
        if not key in time_table: 
            time_table[key] = {
            'first_dt': df_mbp['second'].iloc[i],
            'list': list()
            }
        time_table[key]['list'].append(df_mbp["ART_MBP"].iloc[i])

    # 3-2 안정적 MBP 지속: 1분 평균 MBP 65이상으로 5회 지속 
    str_point = 0
    str_p_cnt = 0    
    for i, (k, v) in enumerate (time_table.items(),0): ## 
        avg_1m = sum(v['list']) / len(v['list']) ## 1분 평균 MBP
        if (avg_1m >= THRESHOLD_E) & (str_p_cnt<=5): ## 65 이상 5회 카운트
            str_p_cnt+=1        
        elif str_p_cnt>5: # 안정적인 경우,
            str_point = k
            break         
        elif k in skip_min_list:  # 25개 미만이면 다시 처음부터
            str_p_cnt =0
        else: # 일시적 안정이므로 다시 첨부터 카운트
            str_p_cnt =0
            
    # 4. 이벤트 추출
    evt_idx = 0
    n_evt_idx = 0
    for i, (k, v) in enumerate (zip(time_table.keys(),time_table.values()),0):
        if k < str_point:
            continue
        if k not in skip_min_list:
            avg_1m = sum(v['list']) / len(v['list']) ## 1분 평균 

            if (avg_1m < THRESHOLD_E): ## 이벤트 기록 
                e_tmp = {
                    'case': c_id, ## case id
                    'event_idx': evt_idx,
                    'minute': k, 
                    'second': list(range(int(v['first_dt']),  int(v['first_dt']) + len(v['list']) *2,2)),
                    'first_dt': v['first_dt'], ## timestamp 같이 저장
                    'mbps' : v['list']
                }
                event_cand.append(e_tmp)
                pd.DataFrame(e_tmp)[['mbps','second']].to_csv(os.path.join(SAVE_ROOT_DIR,"{:05d}_MBP.0M.e.{:03d}.csv".format(c_id,evt_idx)))
                temp_sur_sec = []
                temp_sur_mbp = []
                temp_sur_first= []
                for sur_min in range(-5,5): ## 주변값가져오기
                    try:
                        sample_sur = time_table[k+sur_min]
                        temp_sur_sec.extend(list(range(int(sample_sur['first_dt']),  int(sample_sur['first_dt']) + len(sample_sur['list']) *2,2)))
                        temp_sur_mbp.extend(sample_sur['list'])
                        temp_sur_first.append(sample_sur['first_dt']) 
                    except: pass
                e_surr_tmp = {
                    'case': c_id, ## case id 
                    'event_idx': evt_idx,
                    'minute': k, 
                    'second': temp_sur_sec,
                    #'first_dt': temp_sur_first,
                    'mbps' : temp_sur_mbp,
                }
                event_cand_surround.append(e_surr_tmp)      
                #pd.DataFrame(e_surr_tmp)[['mbps','second']].to_csv(os.path.join(SAVE_ROOT_DIR,"{:05d}_MBP_surr.0M.e.{:03d}.csv".format(case_id,evt_idx)))
                evt_idx+=1
                skip_min_list.extend(list(range(k,k+20))) # omitting time range

            mbps_window.append(avg_1m) # 매분에 대한 MBP 기록 저장

            if(len(mbps_window) > TIME_WINDOW): # 30분 기록 저장 
                mbps_window = mbps_window[1:]

            if(len(mbps_window) == TIME_WINDOW and min(mbps_window) >= THRESHOLD_N): 
                ## 30분 동안 안정적인 MBP 경우
                ## 중간 (15분) 앞뒤로 1분에 대해 총 3분 추출 
                nonevents.append({
                  'case' : c_id,
                    'minute': k, 
                    'event_idx': n_evt_idx,
                  'second' : (k - 15 + 1) * 60, # 현재 minute - 15분 + 1분 * 60초
                  'first_dt': v['first_dt'] + (- 15 + 1) * 60, 
                })
                nonevents.append({
                  'case' : c_id,
                    'minute': k,
                    'event_idx': n_evt_idx,
                  'second' : (k - 15) * 60, # 현재 minute - 30분 * 60초
                  'first_dt': v['first_dt'] + (- 15) * 60, 
                })
                nonevents.append({ 
                  'case' : c_id,
                   'minute': k, 
                    'event_idx': n_evt_idx,
                  'second' : (k - 15 - 1) * 60, # 현재 minute - 15분 - 1분 * 60초
                  'first_dt': v['first_dt'] + (- 15 -1) * 60, 
                })                
                n_evt_idx+=1
                mbps_window = []  ## 기록 초기화   
                
    return event_cand_surround, event_cand, nonevents

def get_wave(event_list,c_id,id_trks, event_tf=True):## False if event is a non-event,    
    #f_waves = id_trks['tname'][(id_trks['type']=='W')] ## wave file list read
    for f in ['BIS/EEG1_WAV', 'SNUADC/ART', 'SNUADC/ECG_II']: #f_waves: #['SNUADC/ART']: #
        wav_f_name = f.replace('/','_')
        npz_path = FILE_ROOT_DIR+'{:05d}'.format(c_id)+'_'+wav_f_name+'.npz' ## file 명
        try: 
            value = np.load(npz_path)['arr_0'] # file read  
        except: 
            print('there is no file:', wav_f_name)
            pass
        else:
            df_wav = pd.DataFrame(value).reset_index() ## index column 추가 --> 이후 time stamp 계산을 위해
            wav_dtstart = id_trks['dtstart'][(id_trks['tname']==f)].values[0] ## wave start 시간 찾기
            wav_dtend = id_trks['dtend'][(id_trks['tname']==f)].values[0]
            wav_hertz = id_trks['srate'][(id_trks['tname']==f)].values[0] ## wave hertz 시간 찾기
            wav_gain = id_trks['gain'][(id_trks['tname']==f)].values[0] ## wave hertz 시간 찾기
            wav_bias = id_trks['bias'][(id_trks['tname']==f)].values[0] ## wave hertz 시간 찾기

            ## 핵심 !! --> 측정 시작 시간의 기록만 존재, 시작 시간으로 부터 각 hertz의 초를 반영하여 value의 측정된 시간 생성
            ## 측정시간은 index(측정순서)/hertz -> 단위: time stamp                   
            df_wav["timestamp"] = wav_dtstart+df_wav['index'].astype(float)/wav_hertz 
            df_wav = df_wav.rename({0:f},axis=1) ## wave value column 이름 변경
            df_wav["second"] = df_wav['timestamp'].apply(lambda x: x - TIME_OFFSET) # 타임 스탬프 변경 --> 초
            df_wav["minute"] = df_wav["second"].apply(lambda x: int(x / 60)) ## 초 --> 분 기록
            df_wav = df_wav.drop('index',axis=1) ## 저장용량 줄이기 위해서 index column 날림
            
            print(f, wav_dtstart,wav_dtend,df_wav["second"].iloc[-1])
            
            if event_tf == False:
                chk_points = [CHK_POINT[0]]
                prefix_f_name = '{:05d}_{}.n'.format(c_id,wav_f_name)
            else:
                chk_points = CHK_POINT
                prefix_f_name = '{:05d}_{}.e'.format(c_id,wav_f_name)
            
            for event in event_list:
                for i, chk_p in enumerate (chk_points,1): ## event case 저장
                    ## event time 발생 prev_point*60초전 시간 get
                    start_point = (event['first_dt'] - (chk_p*60))
                    end_point = (event['first_dt'] - (chk_p*60) +60)                               
                    e_bound = (start_point, end_point)     
                    wave_in_bound = df_wav[[f,'minute','second']][(df_wav['second']>=e_bound[0]) & (df_wav['second']<=e_bound[1])].reset_index(drop=True)              
                    wave_in_bound[f] = wave_in_bound[f]*wav_gain+wav_bias ## 보정

                    ## file 이름: caseid_wavename_minute
                    if len(wave_in_bound)>100:
                        wave_in_bound.to_csv(os.path.join(SAVE_ROOT_DIR,prefix_f_name+".{}M.{:03d}_{:.3f}.csv".format(chk_p,event['event_idx'],end_point)))

                        ## 이벤트 타임 주변시간 저장
                        '''
                        if chk_p == 0:
                            e_bound_surround = (event['first_dt'] - ((chk_p+5)*60)), (event['first_dt'] + ((chk_p+5)*60+60))
                            wave_surround_in_bound = df_wav[[f,'minute','second']][(df_wav['second']>=e_bound_surround[0])& (df_wav['second']<=e_bound_surround[1])].reset_index(drop=True)
                            wave_surround_in_bound[f] = wave_surround_in_bound[f]*wav_gain+wav_bias ## 보정               
                            wave_surround_in_bound.to_csv(os.path.join(SAVE_ROOT_DIR,"{:05d}_{}.{}M.e_surr.csv".format(c_id,wav_f_name,chk_p)))
                          #  return wave_in_bound, wave_surround_in_bound
                        '''
def save_events(case_list):
    for i, case_id in enumerate (case_list,1):
        case_id = int(case_id)
        print(case_id)
        case_info = VITAL_TRKS[(VITAL_TRKS['caseid']==case_id)]
        events_surround, events, nonevents = event_selection(case_id,case_info)
        print('num_events:',len(events), 'num_non_event',len(nonevents))
        if events:
            get_wave(events,case_id,case_info)
        if nonevents:
            get_wave(nonevents,case_id,case_info,event_tf=False) 

if __name__ == '__main__':   
    procs = []
    total_progress = len(CASE_IDS)
    th_job = int(total_progress/THREADS)
    
    for thd_id in range (THREADS):
        job_list = CASE_IDS[thd_id*th_job:(thd_id+1)*th_job]
        if thd_id == THREADS-1:
            job_list = CASE_IDS[thd_id*th_job:]
        proc = Process(target=save_events, args=([job_list]))
        procs.append(proc)
        proc.start() 
    
    for proc in procs:
        proc.join()    

