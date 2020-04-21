
import torch.nn as nn
import torch

class SkipConnection (nn.Module):
    def __init__(self,in_channel,out_channel,keep_dim=True):
        super(SkipConnection,self).__init__()

        if in_channel != out_channel:
            self.conv1d = nn.Conv1d(in_channel,out_channel,1)
        initialize_weights(self)
    def forward(self,before,after):
        '''
        :param before: the tensor before passing convolution blocks
        :param after: the tensor of output from convolution blocks
        :return: the sum of inputs
        '''
        if before.shape[2] != after.shape[2]: # if the length is different (1/2)
            before = nn.functional.max_pool1d(before,2,2)

        if before.shape[1] != after.shape[1]:
            before = self.conv1d(before)
        return before + after
    
class ResidualBlock (nn.Module):
    def __init__(self,in_channel,out_channel,pool=False,kernel_size=15,
                 activation=nn.LeakyReLU(),do=0.5):
        super(ResidualBlock,self).__init__()
        self.pool = pool
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            activation,
            nn.Dropout(do),
            nn.Conv1d(in_channel,out_channel, kernel_size,1,kernel_size // 2),
            nn.BatchNorm1d(out_channel),
            activation,
            nn.Dropout(do),
            nn.Conv1d(out_channel,out_channel, kernel_size,1,kernel_size // 2)
        )
        self.skip = SkipConnection(in_channel,out_channel)
        initialize_weights(self)
        
    def forward(self, input):  
        out = self.block(input)
        if self.pool:
            out = nn.functional.max_pool1d(out,2,2)
        out = self.skip(input,out)

        return out


class ResidualEncoder (nn.Module):
    def __init__(self,seq_length, num_channel,kernel_size,
                 do,activation):
        assert (kernel_size//2)%2, print('kernel_size//2 must be odds')

        super(ResidualEncoder,self).__init__()
        self.num_channel = num_channel
        self.kernel_size = kernel_size
        
        self.conv1st = nn.Sequential(
            nn.Conv1d(1,self.num_channel,self.kernel_size,1,self.kernel_size//2),
            nn.BatchNorm1d(self.num_channel)
        )
        
        self.block_info = [
            # in channel , out channel, on/off max_pooling_mode, kernel_size  풀링은 6번!!!
            (self.num_channel,self.num_channel,1,self.kernel_size),
            (self.num_channel,self.num_channel,0,self.kernel_size),
            (self.num_channel,self.num_channel,1,self.kernel_size),
            (self.num_channel,self.num_channel,0,self.kernel_size),
            (self.num_channel,self.num_channel*2,1,self.kernel_size),
            (self.num_channel*2,self.num_channel*2,0,self.kernel_size),
            (self.num_channel*2,self.num_channel*2,1,self.kernel_size//2),
            (self.num_channel*2,self.num_channel*2,0,self.kernel_size//2),
            (self.num_channel*2,self.num_channel*3,1,self.kernel_size//2),
            (self.num_channel*3,self.num_channel*3,0,self.kernel_size//2),
            (self.num_channel*3,self.num_channel*3,1,self.kernel_size//2),
            (self.num_channel*3,self.num_channel*3,0,self.kernel_size//2),
        ]
        
        self.pool_count = 0
        for ch_info in self.block_info:
            self.pool_count +=ch_info[2]
       
        self.output_size = seq_length
        
        for i in range(self.pool_count):
            self.output_size = self.output_size//2
  
        self.encoder = nn.Sequential()
        for i,(in_ch,out_ch,max_pooling_mode,kernel_size) in enumerate(self.block_info):
            self.encoder.add_module('residual_block_{}'.format(i),
                                    ResidualBlock(in_ch,out_ch,max_pooling_mode,kernel_size,activation,do))
        initialize_weights(self)
        
    def forward(self,input):
        out = self.conv1st(input)
        out = self.encoder(out)
        return out


class ResidualNetHDI(nn.Module): #
    def __init__(self,seq_length,kernel_size,
                 num_channel, 
                 class_num,
                 do=0.5,activation=nn.LeakyReLU()):
        
        super(ResidualNetHDI,self).__init__()
        self.seq_length = seq_length  # the number of points of a wave form 
        self.num_channel = num_channel
        self.kernel_size = kernel_size 
        self.class_num = class_num
        
        self.encoder = ResidualEncoder(self.seq_length,
                                self.num_channel, #channel 
                                   self.kernel_size,
                                   do,activation)
        
        
        self.bn = nn.BatchNorm1d(num_channel*3)
        self.out_size = self.encoder.output_size
        self.flatten_len = self.out_size*self.num_channel*3
        self.linear = torch.nn.Sequential(
            nn.Linear(self.flatten_len,32),
            nn.LeakyReLU(),
            nn.Dropout(do),            
          #  nn.Linear(self.hid_size , self.n_classes), # second layer
         #   nn.Softmax(dim=1), 
        )
       # self.soft_max = nn.Softmax(1)
        initialize_weights(self)
    
    def activations_hook(self,grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self,x):
        out = self.encoder(x)
        return out
        
    def forward(self,x,GRAD=False): # x : a set of lists consisting of wave forms
        out = self.encoder(x)
        
        if GRAD:
            h = out.register_hook(self.activations_hook)
        
        out = self.linear(out.view(-1,self.flatten_len))
        #out = self.soft_max(out)
        return out
    
    
    
class HDIClassifier (nn.Module):
    def __init__(self,wave_list,kernel_list,class_num,num_channel):
        super(HDIClassifier, self).__init__()
        
        self.seq_len_dict ={
                'ecg':30000,
                'abp':30000,
                'eeg':7680
        }

        self.seq_idx_dict ={
                'ecg':[0,30000],
                'abp':[30000,60000],
                'eeg':[60000,67680]
        }

        self.num_wave = len(wave_list)
        self.wave_list = wave_list
        self.model_dict = torch.nn.ModuleDict()
        
        for wavename in self.wave_list:
            model=ResidualNetHDI(self.seq_len_dict[wavename],kernel_list[0],num_channel=num_channel, class_num=2)
            self.model_dict.add_module(wavename,model)
            

        # self.ecg = ResidualNetHDI(seq_len_list[0],kernel_list[0],num_channel=num_channel,class_num=2)
        # self.abp = ResidualNetHDI(seq_len_list[1],kernel_list[1],num_channel=num_channel,class_num=2)
        # self.eeg = ResidualNetHDI(seq_len_list[2],kernel_list[2],num_channel=num_channel,class_num=2)

        self.mlp = torch.nn.Sequential(
            nn.Linear(32*self.num_wave,16),
            nn.Linear(16 , class_num), 
            nn.Softmax(dim=1), 
        )
        initialize_weights(self)
         
    def forward(self,x,GRAD=False): 
        
        output_list = []
        for wavename in self.wave_list:
            out = self.model_dict[wavename](x[:,self.seq_idx_dict[wavename][0]:self.seq_idx_dict[wavename][1]].view(-1,1,self.seq_len_dict[wavename]),GRAD)
            output_list.append(out)
        
        # out0 = self.ecg(x[:,:self.seq_len_list[0]].view(-1,1,self.seq_len_list[0]),GRAD)
        # out1 = self.abp(x[:,self.seq_len_list[0]:self.seq_len_list[0]+self.seq_len_list[1]].view(-1,1,self.seq_len_list[1]),GRAD)
        # out2 = self.eeg(x[:,-self.seq_len_list[2]:].view(-1,1,self.seq_len_list[2]),GRAD)
        output = torch.cat(output_list,1)
        y_hat = self.mlp(output)
        
        return y_hat

def initialize_weights(net):
    torch.manual_seed(1)
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()