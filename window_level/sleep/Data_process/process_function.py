import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(current_path)[0]
root_path = "../../datasets/downstream"

sys.path.append(current_path)

import LoadData
import numpy as np
import scipy.linalg
import scipy.io
import scipy.sparse
import scipy.signal as signal
from braindecode.preprocessing import exponential_moving_standardize
# from einops import rearrange
from sklearn.model_selection import train_test_split


def EMS(data):
    new_x = []

    for x in data:
        new_x.append(exponential_moving_standardize(x))

    return np.array(new_x)


def Load_BCIC_2a_raw_data(tmin=0, tmax=4,bandpass = [0,38],resample = None):
    '''
    Load the BCIC 2a data.
    Arg:
        sub:The subject whose data need to be load.
    '''

    data_path = os.path.join(root_path,'Raw_data','BCICIV_2a_gdf') 
    for sub in range(1,10):
        data_name = r'A0{}T.gdf'.format(sub)
        data_loader = LoadData.LoadBCIC(data_name, data_path)
        data = data_loader.get_epochs(tmin=tmin, tmax=tmax,bandpass = bandpass,resample = resample)
        print('orgin size is:',data['x_data'].shape)
        train_x = np.array(data['x_data'])[:, :, :]
        train_y = np.array(data['y_labels'])
        
        data_name = r'A0{}E.gdf'.format(sub)
        label_name = r'A0{}E.mat'.format(sub)
        data_loader = LoadData.LoadBCIC_E(data_name, label_name, data_path)
        data = data_loader.get_epochs(tmin=tmin, tmax=tmax,bandpass = bandpass,resample = resample)
        test_x = np.array(data['x_data'])[:, :, :]
        test_y = data['y_labels']
        

        train_x = np.array(train_x)
        print(train_x.shape)
        train_x = EMS(train_x)
        print(train_x.shape)
        train_y = np.array(train_y).reshape(-1)

        
        test_x = np.array(test_x)
        
        test_x = EMS(test_x)
        test_y = np.array(test_y).reshape(-1)
 
        print('trian_x:',train_x.shape)
        print('train_y:',train_y.shape)
        
        print('test_x:',test_x.shape)
        print('test_y:',test_y.shape)
        
        if bandpass is None:
            SAVE_path = os.path.join(root_path,'Data','BCIC_2a')
        else:
            SAVE_path = os.path.join(root_path,'Data','BCIC_2a_{}_{}HZ'.format(bandpass[0],bandpass[1]))

        if not os.path.exists(SAVE_path):
            os.makedirs(SAVE_path)
            
        SAVE_test = os.path.join(SAVE_path,r'sub{}_test'.format(sub))
        SAVE_train = os.path.join(SAVE_path,'sub{}_train'.format(sub))
        
        if not os.path.exists(SAVE_test):
            os.makedirs(SAVE_test)
        if not os.path.exists(SAVE_train):
            os.makedirs(SAVE_train)
            
        scipy.io.savemat(os.path.join(SAVE_train, "Data.mat"), {'x_data': train_x,'y_data': train_y})
        scipy.io.savemat(os.path.join(SAVE_test, "Data.mat"), {'x_data': test_x, 'y_data': test_y})
        print('数据保存成功！')

def Load_BCIC_2b_raw_data(tmin=0, tmax=4,bandpass = [0,38]):
    '''
    Load all the 9 subjects data,and save it in the fold of r'./Data'.
    '''
    
    data_path = os.path.join(root_path,'Raw_data','BCICIV_2b_gdf')
    if bandpass is None:
        save_path = os.path.join(root_path,r'Data','BCIC_2b')
    else:
        save_path = os.path.join(root_path,r'Data','BCIC_2b_{}_{}HZ'.format(bandpass[0],bandpass[1]))

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    
    for sub in range(1,10):
        load_raw_data = LoadData.LoadBCIC_2b(data_path,sub,tmin, tmax,bandpass)
        save_train_path = os.path.join(save_path,r'sub{}_train'.format(sub))
        save_test_path = os.path.join(save_path,r'sub{}_test').format(sub)
        if not os.path.exists(save_train_path):
            os.makedirs(save_train_path)
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)
        
        train_x,train_y = load_raw_data.get_train_data()
        scipy.io.savemat(os.path.join(save_train_path,'Data.mat'),{'x_data':train_x,'y_data':train_y})
        
        test_x,test_y = load_raw_data.get_test_data()
        scipy.io.savemat(os.path.join(save_test_path,'Data.mat'),{'x_data':test_x,'y_data':test_y})
        
    print('保存成功！')
    
if __name__ == '__main__':
    
    Load_BCIC_2a_raw_data()
    Load_BCIC_2b_raw_data()
    