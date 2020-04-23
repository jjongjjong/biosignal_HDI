from scipy.signal import butter, lfilter
import torch
import torch.nn.functional as F
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 
import pathlib
import datetime
import os
import numpy as np


def make_save_folder(config):
    now = datetime.datetime.now()
    time_str = datetime.datetime.strftime(now,'%Y%m%d_%H%M%S')
    folder_name = '{}M_{}_{}'.format(config['time_point'],time_str,'_'.join(config['waves']))
    save_path = os.path.join(config['save_path'],folder_name)
    pathlib.Path(save_path).mkdir(exist_ok=True,parents=True)
    pathlib.Path(os.path.join(save_path,'models')).mkdir(exist_ok=True,parents=True)
    return save_path


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


 
def Attention_viewer(model,input_data,signal_name,label,ref_list=None):
    model.eval()
    
    pred_prob = model(input_data,GRAD=True)
    pred_class = pred_prob.argmax(dim=1)
    
    pred_prob[:,pred_class].backward()
    
    if signal_name=='ecg':
        input_len = 30000
        single_data = input_data[:,:30000].unsqueeze(0)
        gradients = model.ecg.get_gradient()
        pooled_gradients = torch.mean(gradients,dim=[0,2])
        activations = model.ecg.get_activations(single_data).detach().cpu()
    elif signal_name =='abp':
        input_len = 30000
        single_data = input_data[:,30000:60000].unsqueeze(0)
        gradients = model.abp.get_gradient()
        pooled_gradients = torch.mean(gradients,dim=[0,2])
        activations = model.abp.get_activations(single_data).detach().cpu()
    elif signal_name =='eeg':
        input_len = 7680
        single_data = input_data[:,-7680:].unsqueeze(0)
        gradients = model.eeg.get_gradient()
        pooled_gradients = torch.mean(gradients,dim=[0,2])
        activations = model.eeg.get_activations(single_data).detach().cpu()

    for i in range(activations.shape[1]):
        activations[:,i,:] *= pooled_gradients[i]

    heatmap = torch.mean(activations,dim=1)#.squeeze()
    heatmap = np.maximum(heatmap,0)
    heatmap /=torch.max(heatmap)
    
    heatmap = cv2.resize(heatmap.numpy(),(input_len,1))
    heatmaps = np.concatenate([heatmap for i in range(10)])
    
    label_name = ref_list[label.item()]
    pred_name = ref_list[pred_class.item()]
    
    #plt.title('real_label:{} pred_label:{}'.format(label_name,pred_name))
    plt.rcParams["figure.figsize"] = 60,20
    fig,ax = plt.subplots(sharex=True)
    ax.imshow(heatmaps,cmap='Blues',aspect='auto')
    ax.title.set_text('real_label:{}    pred_label:{}'.format(label_name,pred_name))
    ax.title.set_fontsize(50)
    ax2 = plt.twinx()
    sns.lineplot(data=single_data.detach().cpu().numpy()[0][0,:],ax=ax2,linewidth=8,color='limegreen' )
    ax.axis('tight')
    plt.show()
    