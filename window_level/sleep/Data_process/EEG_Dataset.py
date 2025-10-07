import os
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import random

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from Data_process.utils import train_validation_split,EA,few_shot_data

class eeg_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,domain = None,domain_label =  False):
        super(eeg_dataset,self).__init__()
        self.domain_label = domain_label
        self.feature = feature
        self.label = label
        self.domain = domain
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.domain_label:
            return self.feature[index], self.label[index],self.domain[index]
        else:
            return self.feature[index], self.label[index]

class mix_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    
    def __init__(self,feature,label,domain = None):
        super(mix_dataset,self).__init__()
        self.feature = feature
        self.label = label

        self.all_feature = feature
        self.all_label = label

        self.domain = domain

        self.train_flag = False
        self.lam = None
    
    def __len__(self):
        return len(self.label)

    def train_setting(self,x,y):
        self.x,self.y = x,y
        
    def train(self):
        self.train_flag = True
        
    def test(self):
        self.train_flag = False
        
    def set_lam(self,lam):
        self.lam = lam
    
    def __getitem__(self, index):
        if self.train_flag:
            label_0 = torch.zeros(4)
            label_0[self.label[index]] = 1

            mixup_idx_0 = random.randint(0,len(self.y)-1)

            mixup_idx_s = random.randint(0,len(self.label)-1)
            
            label_1 = torch.zeros(4)
            label_1[self.label[mixup_idx_s]]=1

            # mixup_idx = self.label[index]
            # beta = np.random.beta(0.2,0.2)

            mixup_label = torch.zeros(4)
            # mixup_label[self.y[mixup_idx_0]]=(1-beta)
            # mixup_label[self.y[mixup_idx_1]]= beta
            # mixup_feature = (1-beta)*self.x[mixup_idx_0]+beta*self.x[mixup_idx_1]

            mixup_label[self.y[mixup_idx_0]] = 1
            mixup_feature = self.x[mixup_idx_0]

            if self.lam is None:
                alpha = 0.2
                lam = np.random.beta(alpha,alpha)
                beta = np.random.beta(alpha,alpha)
                # lam = 1- 0.5*np.random.rand()
            else:
                lam = self.lam
            
            source_mix_x = (1-beta)*self.feature[index] + beta*self.feature[mixup_idx_s]
            source_mix_y = (1-beta)*label_0 + beta*label_1

            idx_feature = (1-lam)*source_mix_x + lam * mixup_feature

            idx_label = (1-lam) *source_mix_y + lam * mixup_label

            domain_label = torch.zeros(2)
            
            #为了能够使用对抗神经网络
            domain_label[0] = 1-lam
            domain_label[1] = lam
            
            if self.domain is not None:
                return idx_feature,idx_label,domain_label,lam
            else:
                return idx_feature,idx_label,lam
        else:
            if self.domain is not None:
                return self.feature[index],self.label[index],self.domain[index]
            else:
                return self.feature[index],self.label[index]
    

def get_test_EEG_data(sub,data_path):
    '''
    Return one subject's test dataset.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
    @author:WenChao Liu 
    '''
    test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))
    test_data = sio.loadmat(test_path)
    test_x = test_data['x_data']
    test_y = test_data['y_data']
    test_x,test_y = torch.FloatTensor(test_x),torch.LongTensor(test_y).reshape(-1)
    test_dataset = eeg_dataset(test_x,test_y)
    return test_dataset

def get_HO_EEG_data(sub,data_path,validation_size=0.2,data_seed=20210902):
    
    '''
    Return one subject's training dataset,split training dataset and split validation dataset.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        validation_rate:The percentage of validation data in the data to be divided. 
        data_seed:The random seed for shuffle the data.
    @author:WenChao Liu
    '''
    train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
   
    train_data = sio.loadmat(train_path)
    train_x =  train_data['x_data']
    train_y = train_data['y_data'].reshape(-1)
    print(train_x.shape,train_y.shape)
        
    split_train_x,split_train_y,split_validation_x,split_validation_y = train_validation_split(train_x,train_y,validation_size,seed=data_seed)
    
    train_x,train_y = torch.FloatTensor(train_x),torch.LongTensor(train_y).reshape(-1)
    split_train_x,split_train_y = torch.FloatTensor(split_train_x),torch.LongTensor(split_train_y).reshape(-1)
    split_validation_x,split_validation_y = torch.FloatTensor(split_validation_x),torch.LongTensor(split_validation_y).reshape(-1)
   
    train_dataset = eeg_dataset(train_x,train_y)
    split_train_dataset = eeg_dataset(split_train_x,split_train_y)
    split_validation_dataset = eeg_dataset(split_validation_x,split_validation_y)    
    test_dataset = get_test_EEG_data(sub,data_path)
    
    return train_dataset,split_train_dataset,split_validation_dataset,test_dataset


def get_CV_EEG_data(sub,data_path,k=10,validation_size=0.2,data_seed=20210902,all_session = False):
    '''
    Get the data in KFCV. 
    Arg:
        sub: Subject number.
        data_path: The data  path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        k: K folds cross validation. 
        validation_size:The percentage of validation data in the training data to be divided.
        data_seed:To shuffel the data in the function:train_validation_split.
    Return: A generator to get the kfcv data.
    '''
    path = os.path.join(data_path,'sub{}_train'.format(sub),'Data.mat')
    data = sio.loadmat(path)
    
    data_x = data['x_data']
    data_y = data['y_data'].reshape(-1)
    
    
    if all_session:
        session_2_path = os.path.join(data_path,r'sub{}_test'.format(sub),'Data.mat')
        session_2_data = sio.loadmat(session_2_path)
        session_2_x = session_2_data['x_data']
        session_2_y = session_2_data['y_data'].reshape(-1)
        print(data_x.shape)
        print(session_2_x.shape)
        data_x = np.concatenate((data_x,session_2_x))
        data_y = np.concatenate((data_y,session_2_y))
        
    skf = StratifiedKFold(n_splits=k,shuffle=True,random_state= data_seed)
    
    for train_index,test_index in skf.split(data_x,data_y):
        train_x = data_x[train_index]
        train_y = data_y[train_index]
        test_x = data_x[test_index]
        test_y = data_y[test_index]
        print(train_x.shape)
        print(train_y.shape)
        
        split_train_x,split_train_y,split_validation_x,split_validation_y = train_validation_split(train_x,train_y,validation_size,seed=data_seed)
        
        train_x = torch.FloatTensor(train_x)
        train_y = torch.LongTensor(train_y)
        test_x = torch.FloatTensor(test_x)
        test_y = torch.LongTensor(test_y)
        split_train_x = torch.FloatTensor(split_train_x)
        split_train_y = torch.LongTensor(split_train_y)
        split_validation_x = torch.FloatTensor(split_validation_x)
        split_validation_y = torch.LongTensor(split_validation_y)
        
        yield eeg_dataset(train_x,train_y),eeg_dataset(split_train_x,split_train_y),eeg_dataset(split_validation_x,split_validation_y),eeg_dataset(test_x,test_y)

def get_HOCV_EEG_data(sub,data_path,k=5,data_seed=20210902):
    
    '''
    This version dosen't use early stoping.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        validation_rate:The percentage of validation data in the data to be divided. 
        data_seed:The random seed for shuffle the data.
    @author:WenChao Liu
    '''
    train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
   
    train_data = sio.loadmat(train_path)
    train_x =  train_data['x_data']
    train_y = train_data['y_data'].reshape(-1)
    print(train_x.shape,train_y.shape)


    skf =   StratifiedKFold(n_splits=k,shuffle=True,random_state= data_seed)
    for split_train_index,split_validation_index in skf.split(train_x,train_y):
        split_train_x = train_x[split_train_index]
        split_train_y = train_y[split_train_index]
        split_validation_x =train_x[split_validation_index]
        split_validation_y = train_y[split_validation_index]

    
        train_x,train_y = torch.FloatTensor(train_x),torch.LongTensor(train_y).reshape(-1)
        split_train_x,split_train_y = torch.FloatTensor(split_train_x),torch.LongTensor(split_train_y).reshape(-1)
        split_validation_x,split_validation_y = torch.FloatTensor(split_validation_x),torch.LongTensor(split_validation_y).reshape(-1)
   
        split_train_dataset = eeg_dataset(split_train_x,split_train_y)
        split_validation_dataset = eeg_dataset(split_validation_x,split_validation_y)    
        # test_dataset = get_test_EEG_data(sub,data_path)
    
        yield split_train_dataset,split_validation_dataset


def get_CSE_unsupervised_data(sub,data_path,Tr_size = 0.25,Eu_ai = False,random = 31415926):
    '''
    Get the unsupercised dataset of one subject.Note that all of the session 1 data is taken as the training data for source domain ,and session 2 data is splited into training and test parts for the target domain.
    Arg:
        sub:Subject number.
        data_poath:The data path of all subjects.
        augment:Data augmention.Take care that this operaton will change the size in the temporal dimension.
    '''
    source_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))
    source_data = sio.loadmat(source_path)

    target_data = sio.loadmat(target_path)

    target_feature = target_data['x_data']
    target_label = target_data['y_data'].reshape(-1)


    target_train_x,target_test_x,target_train_y,target_test_y  = train_test_split(target_feature,target_label,train_size = Tr_size,random_state = random,stratify=target_label)
    
    target_train_x = torch.FloatTensor(target_train_x)
    target_train_y = torch.LongTensor(target_train_y)
    
    target_test_x = torch.FloatTensor(target_test_x)
    target_test_y = torch.LongTensor(target_test_y)

    # source_feature = source_data['x_data']
    if Eu_ai:
        xt = np.transpose(target_train_x.numpy(),axes=(0,2,1))
        print('xt shape:',xt.shape)
        E = np.matmul(target_train_x.numpy(),xt)
        print(E.shape)
        R = np.mean(E, axis=0)
        source_feature = EA(source_data['x_data'],R)
    else:
        source_feature = source_data['x_data']
    
    source_label = source_data['y_data'].reshape(-1)  

    
    source_feature = torch.FloatTensor(source_feature)
    source_label = torch.LongTensor(source_label).reshape(-1)
    
    return eeg_dataset(source_feature,source_label),eeg_dataset(target_train_x,target_train_y),eeg_dataset(target_test_x,target_test_y)

def get_CSU_EEG_data(sub,data_path,use_all_source = False,split = False,target_mix = False,few_shot = False,few_shot_number = None):
            
    '''
    获得用于跨被试的数据集，该被试的Session1用作验证集，Session2用作测试集。其他被试用作训练集。
    Arg:
       sub:Subject number
       data_path:The data path of all data.
       use_all_source:Use both the session1 and session2 data of the source subjects.
       split_target_train:Whether to split the target train data.
    '''  
    
    target_y_data = []
    
    target_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    val_test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_path)
    session_2_data = sio.loadmat(val_test_path)
   
    
    if target_mix:
        target_x_data = []
        target_x_data.extend(session_1_data['x_data'])
        target_x_data.extend(session_2_data['x_data'])
        
        target_x_data = torch.FloatTensor(np.array(target_x_data))
    
        target_y_data = torch.LongTensor(np.array([session_1_data['y_data'],session_2_data['y_data']])).reshape(-1)

        # print(target_x_data.shape,target_y_data.shape)

        target_x,test_x,target_y,test_y = train_test_split(target_x_data,target_y_data,train_size=0.5, stratify= target_y_data)
    else:
        if few_shot:
            target_x,target_y = few_shot_data(sub,data_path,4,few_shot_number)
            target_x,target_y = torch.FloatTensor(target_x),torch.LongTensor(target_y)
        else:    
            target_x = torch.FloatTensor(session_1_data['x_data'])
                
            target_y = torch.LongTensor(session_1_data['y_data']).reshape(-1)

        test_x = torch.FloatTensor(session_2_data['x_data'])      
        test_y = torch.LongTensor(session_2_data['y_data']).reshape(-1)


    target_train_domain = torch.ones(len(target_y)).reshape(-1).long()
    target_test_domain = torch.ones(len(test_y)).reshape(-1).long()

    target_train_dataset = mix_dataset(target_x,target_y,target_train_domain)

    target_test_dataset = mix_dataset(test_x,test_y,target_test_domain)


    source_x = []
    source_y = []
    #  #算一下目标域整体的协方差
    xt = np.transpose(target_x.numpy(),axes=(0,2,1))
    print('xt shape:',xt.shape)
    E = np.matmul(target_x.numpy(),xt)
    print(E.shape)
    R = np.mean(E, axis=0)

    for i in range(1,10):
        if i == sub:
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
        
        if use_all_source:
            test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
            test_data = sio.loadmat(test_path)

            source_x.extend(EA(test_data['x_data'],R))
            source_y.extend(test_data['y_data'].reshape(-1))
        
        
        # print(train_data['y_data'].shape)
        source_x.extend(EA(train_data['x_data'],R))
        source_y.extend(train_data['y_data'].reshape(-1))
    

    source_x = np.array(source_x)
    source_y = np.array(source_y)

    train_x = torch.FloatTensor(source_x)
    train_y = torch.LongTensor(source_y).reshape(-1)
    
    source_domain = torch.zeros(len(train_y)).reshape(-1).long()
    source_dataset = mix_dataset(train_x,train_y,source_domain)
    
    return source_dataset,target_train_dataset,target_test_dataset




if __name__ == '__main__':
    
    # Test function:get_CV_EEG_data.
    path = os.path.join(root_path,'Data','BCIC_2a')
    
    # tr_,ta_,v_,t_ = get_CSU_selected_sub(1,[3,4,6,7],path)
    
    # tr_,ta_,v_,t_ = get_CSU_EEG_data(1,path,False,True,False,True)
    # source,target_train,target_validation,target_test = get_CSE_data(1,path,False,3,False)
    
    for source,target_train,target_validation,target_test in get_CV_EEG_data(1,path,10,0.2,all_session=True):
    
        print(len(source),len(target_train),len(target_validation),len(target_test))
    
    # print(tr_.feature.shape,ta_.feature.shape,v_.feature.shape,t_.feature.shape)
    # print(tr_.label.shape,ta_.label.shape,v_.label.shape,t_.label.shape)
    # unique_class = torch.unique(v_.label)
    # for i in unique_class:
    #     n = torch.sum(v_.label == i)
    #     print('the {}:{}'.format(i,n))
