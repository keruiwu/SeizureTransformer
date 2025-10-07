import numpy as np
import scipy
import os
import scipy.io as sio

def train_validation_split(x,y,validation_size,seed = None):
    '''
    Split the training set into a new training set and a validation set
    @author: WenChao Liu
    '''
    if seed:
        np.random.seed(seed)
    label_unique = np.unique(y)
    validation_x = []
    validation_y = []
    train_x = []
    train_y = []
    for label in label_unique:
        index = (y==label)
        label_num = np.sum(index)
        print("class-{}:{}".format(label,label_num))
        class_data_x = x[index]
        class_data_y = y[index]
        rand_order = np.random.permutation(label_num)
        class_data_x,class_data_y = class_data_x[rand_order],class_data_y[rand_order]
        print(class_data_x.shape)
        validation_x.extend(class_data_x[:int(label_num*validation_size)].tolist())
        validation_y.extend(class_data_y[:int(label_num*validation_size)].tolist())
        train_x.extend(class_data_x[int(label_num*validation_size):].tolist())
        train_y.extend(class_data_y[int(label_num*validation_size):].tolist())
    
    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y).reshape(-1)
    
    train_x = np.array(train_x)
    train_y = np.array(train_y).reshape(-1)
    
    print(train_x.shape,train_y.shape)
    print(validation_x.shape,validation_y.shape)
    return train_x,train_y,validation_x,validation_y

# 欧氏空间的对齐方式 其中x：NxCxS
def EA(x,new_R = None):
    # print(x.shape)
    '''
    The Eulidean space alignment approach for EEG data.

    Arg:
        x:The input data,shape of NxCxS
        new_R：The reference matrix.
    Return:
        The aligned data.
    '''
    
    xt = np.transpose(x,axes=(0,2,1))
    # print('xt shape:',xt.shape)
    E = np.matmul(x,xt)
    # print(E.shape)
    R = np.mean(E, axis=0)
    # print('R shape:',R.shape)

    R_mat = scipy.linalg.fractional_matrix_power(R,-0.5)
    new_x = np.einsum('n c s,r c -> n r s',x,R_mat)
    if new_R is None:
        return new_x

    new_x = np.einsum('n c s,r c -> n r s',new_x,scipy.linalg.fractional_matrix_power(new_R,0.5))
    
    return new_x


def few_shot_data(sub,data_path, class_number = 4,shot_number = 1):
    
    sub_path = os.path.join(data_path,'sub{}_train'.format(sub),'Data.mat')
    data = sio.loadmat(sub_path)
    x,y = data['x_data'],data['y_data'].reshape(-1)
    result_x = []
    result_y = []
    for i in range(class_number):
        label_index = (y == i)
        result_x.extend(x[label_index][:shot_number])
        result_y.extend([i]*shot_number)
        
    return np.array(result_x),np.array(result_y)    

