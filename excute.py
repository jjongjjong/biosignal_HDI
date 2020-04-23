import torch
from model import HDIClassifier
from utils import FocalLoss,make_save_folder
import pathlib
import os
from prepare_dataset import prepare_dataset
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,average_precision_score
import datetime
import pandas as pd
import math
import shutil

def excute(config):
    torch.cuda.empty_cache()
    torch.manual_seed(1)

    model = HDIClassifier(config['waves'],config['windows'],config['channel_n'],2)
    device = torch.device(config['device']) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(),lr = 0.0005)
    loss_f = torch.nn.CrossEntropyLoss() #FocalLoss(logits=True)
    epochs = 15
    batch_size = 270*(int(math.ceil(3/len(config['waves']))))
    


    tr_dataloader,vd_dataloader,te_dataloader = prepare_dataset(config['time_point'],batch_size,config['sqi'])

    loss_list = []
    result_df_list = []

    best_auroc = 0
    best_model_path = None
    
    save_path = make_save_folder(config)
    print(save_path)
    shutil.copytree('./',os.path.join(save_path,'pyfile'))
    #/ home / jjong / jjong / workplace / datathon_2019 / pyfile
    print('Settings: {}M_{}'.format(config['time_point'],'_'.join(config['waves'])))
    for epoch in range(epochs):
        model.train()
        total_loss=0
        
        tr_pred_digit = []
        tr_pred_prob = []
        tr_target_digit = []
        
        for idx,(X,y) in enumerate(tr_dataloader):
            X,y = X.to(device),y.to(device).long()
            optim.zero_grad()
            
            output = model(X)
            
            #loss_f(output[:,1],tmp[1].squeeze(1).to(device))
            loss = loss_f(output,y.squeeze(1).to(device))
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            if idx%50==0:
                print('[{}epoch][{}/{}iter][loss:{}]'.format(epoch,idx,len(tr_dataloader),loss.item()))
            
            tr_target_digit.extend(y.cpu().numpy().ravel().tolist())
            tr_pred_digit.extend(output.max(dim=1)[1].cpu().numpy().tolist())
            tr_pred_prob.extend(output[:,1].detach().cpu().numpy().tolist())
            
        else: 
            loss_list.append(total_loss/len(tr_dataloader))
            print('-----------<[{} epoch] Train Result>----------------'.format(epoch))
            print('Settings: {}M_{}'.format(config['time_point'],'_'.join(config['waves'])))
            print('Train total: [loss:{}]'.format(total_loss/len(tr_dataloader)))
            auroc = roc_auc_score(tr_target_digit,tr_pred_prob)
            auprc = average_precision_score(tr_target_digit,tr_pred_prob)
            print('AUROC : {}'.format(auroc))
            print('AUPRC : {}'.format(auprc))
            print(classification_report(tr_target_digit, tr_pred_digit, labels=[0,1],target_names=['normal','Event']))    
            
            report_dict = classification_report(tr_target_digit, tr_pred_digit, labels=[0,1],target_names=['normal','Event'],output_dict=True)
            report_df = pd.DataFrame(report_dict)
            report_df['epoch'] = epoch
            report_df['state'] = 'train'
            report_df['auroc'] = auroc
            report_df['auprc'] = auprc
            
            result_df_list.append(report_df)
            
        '''----Validation----'''
        with torch.no_grad():
            vd_target_digit=[]
            vd_pred_digit=[]
            vd_pred_prob = []
            
            model.eval()
            for idx,(X,y) in enumerate(vd_dataloader):
                X,y = X.to(device),y.to(device)

                output = model(X)
                vd_target_digit.extend(y.cpu().numpy().ravel().tolist())
                vd_pred_digit.extend(output.max(dim=1)[1].cpu().numpy().tolist())
                vd_pred_prob.extend(output[:,1].detach().cpu().numpy().tolist())
                
            else:
                print('-----------<[{} epoch] Valid Result>----------------'.format(epoch))
                print('Settings: {}M_{}'.format(config['time_point'],'_'.join(config['waves'])))
                auroc = roc_auc_score(vd_target_digit,vd_pred_prob)
                auprc = average_precision_score(vd_target_digit,vd_pred_prob)
                print('AUROC : {}'.format(auroc))
                print('AUPRC : {}'.format(auprc))
                print(classification_report(vd_target_digit, vd_pred_digit, labels=[0,1],target_names=['normal','Event']))    
                report_dict = classification_report(vd_target_digit, vd_pred_digit, labels=[0,1],target_names=['normal','Event'],output_dict=True)
                report_df = pd.DataFrame(report_dict)
                report_df['epoch'] = epoch
                report_df['state'] = 'valid'
                report_df['auroc'] = auroc
                report_df['auprc'] = auprc
                result_df_list.append(report_df)
                pd.concat(result_df_list,sort=True).to_csv(os.path.join(save_path,'result_df.csv'))
                
                if auroc>best_auroc:
                    print('SAVED')
                    best_auroc = auroc
                    best_model_path = os.path.join(save_path,'models','{}_{:.3f}.pth'.format(epoch,auroc))
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'auroc': auroc,
                    }, best_model_path )

    '''----Test----'''
    with torch.no_grad():
            te_target_digit=[]
            te_pred_digit=[]
            te_pred_prob=[]
            
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            model.eval()
            for idx,(X,y) in enumerate(te_dataloader):
                X,y = X.to(device),y.to(device)

                output = model(X)
                te_target_digit.extend(y.cpu().numpy().ravel().tolist())
                te_pred_digit.extend(output.max(dim=1)[1].cpu().numpy().tolist())
                te_pred_prob.extend(output[:,1].detach().cpu().numpy().tolist())
            else:
                print('-----------< Test Result >----------------')
                print('[setting]: {}M_{}'.format(config['time_point'],'_'.join(config['waves'])))
                print('[Best Model]: ',best_model_path)
                auroc = roc_auc_score(te_target_digit,te_pred_prob)
                auprc = average_precision_score(te_target_digit,te_pred_prob)
                print('AUROC : {}'.format(auroc))
                print('AUPRC : {}'.format(auprc))
                print(classification_report(te_target_digit, te_pred_digit, labels=[0,1],target_names=['normal','Event']))    
        
                report_dict = classification_report(te_target_digit, te_pred_digit, labels=[0,1],target_names=['normal','Event'],output_dict=True)
                report_df = pd.DataFrame(report_dict)
                report_df['epoch'] = epoch
                report_df['state'] = 'test'
                report_df['auroc'] = auroc
                report_df['auprc'] = auprc
                result_df_list.append(report_df)
                
                f1_score = report_dict['weighted avg']['f1-score']
                recall = report_dict['weighted avg']['recall']
                precision = report_dict['weighted avg']['precision']
                pd.concat(result_df_list,sort=True).to_csv(os.path.join(save_path,'result_df.csv'))
                with open(os.path.join(save_path,'[{} {:.4f}]_[{}{:.4f}]_[{}{:.4f}]_[{}{:.4f}]_[{}{:.4f}].txt'.format('auroc',auroc,'auprc',auprc,'f1-score',f1_score,'recall',recall,'precision',precision)),'w') as f:
                    f.write('  ')
    del(model)