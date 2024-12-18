import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

def data_loading(dataset_name, augment, load_val_or_test):
    dataset = dataset_name

    data = HeteroData()
    data_neg = HeteroData()
    data_neutral = HeteroData()

    node_types = ['user', 'item']
    attr_names = ['edge_index', 'edge_label_index']

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    if augment == 0:
        df_train = pd.read_csv('dataset/'+dataset+'/'+dataset+'_train_original.csv')
    else:
        df_train = pd.read_csv('dataset/'+dataset+'/'+dataset+'_train_augment.csv')
    df_validation = pd.read_csv('dataset/'+dataset+'/'+dataset+'_validation.csv')
    df_test = pd.read_csv('dataset/'+dataset+'/'+dataset+'_test.csv')

    df_train_neutral = df_train[(df_train['rating:float']==2) | (df_train['rating:float']==3)]
    #df_train_neg = df_train[df_train['rating:float'] < 4]
    df_train_neg = df_train[df_train['rating:float'] <= 1]
    df_train = df_train[df_train['rating:float'] >= 4]
    df_validation = df_validation[df_validation['rating:float'] >= 4]
    df_test = df_test[df_test['rating:float'] >= 4]

    df_train['user_id:token'] = user_encoder.fit_transform(df_train['user_id:token'].values)
    df_train['item_id:token'] = item_encoder.fit_transform(df_train['item_id:token'].values)
    df_train_neutral.loc[:, 'user_id:token'] = user_encoder.transform(df_train_neutral['user_id:token'].values)
    df_train_neutral.loc[:, 'item_id:token'] = item_encoder.transform(df_train_neutral['item_id:token'].values)
    df_train_neg.loc[:, 'user_id:token'] = user_encoder.transform(df_train_neg['user_id:token'].values)
    df_train_neg.loc[:, 'item_id:token'] = item_encoder.transform(df_train_neg['item_id:token'].values)
    #df_train_neg['user_id:token'] = user_encoder.transform(df_train_neg['user_id:token'].values)
    #df_train_neg['item_id:token'] = item_encoder.transform(df_train_neg['item_id:token'].values)
    df_validation['user_id:token'] = user_encoder.transform(df_validation['user_id:token'].values)
    df_validation['item_id:token'] = item_encoder.transform(df_validation['item_id:token'].values)
    df_test['user_id:token'] = user_encoder.transform(df_test['user_id:token'].values)
    df_test['item_id:token'] = item_encoder.transform(df_test['item_id:token'].values)

    if df_train_neg['user_id:token'].dtype != 'int64':
        df_train_neg['user_id:token'] = df_train_neg['user_id:token'].astype(int)
    if df_train_neg['item_id:token'].dtype != 'int64':
        df_train_neg['item_id:token'] = df_train_neg['item_id:token'].astype(int)

    if df_train_neutral['user_id:token'].dtype != 'int64':
        df_train_neutral['user_id:token'] = df_train_neutral['user_id:token'].astype(int)
    if df_train_neutral['item_id:token'].dtype != 'int64':
        df_train_neutral['item_id:token'] = df_train_neutral['item_id:token'].astype(int)

    data[node_types[0]].num_nodes = len(np.unique(df_train['user_id:token'].values))
    data[node_types[1]].num_nodes = len(np.unique(df_train['item_id:token'].values))

    data_neg[node_types[0]].num_nodes = len(np.unique(df_train['user_id:token'].values))
    data_neg[node_types[1]].num_nodes = len(np.unique(df_train['item_id:token'].values))

    data_neutral[node_types[0]].num_nodes = len(np.unique(df_train['user_id:token'].values))
    data_neutral[node_types[1]].num_nodes = len(np.unique(df_train['item_id:token'].values))

    # train
    edge_index = torch.tensor(np.stack([df_train['user_id:token'].values, df_train['item_id:token'].values]))
    data['user', 'rates', 'item'][attr_names[0]] = edge_index
    data['item', 'rated_by', 'user'][attr_names[0]] = edge_index.flip([0])

    edge_index_neg = torch.tensor(np.stack([df_train_neg['user_id:token'].values, df_train_neg['item_id:token'].values]))
    data_neg['user', 'rates', 'item'][attr_names[0]] = edge_index_neg
    data_neg['item', 'rated_by', 'user'][attr_names[0]] = edge_index_neg.flip([0])

    edge_index_neutral = torch.tensor(np.stack([df_train_neutral['user_id:token'].values, df_train_neutral['item_id:token'].values]))
    data_neutral['user', 'rates', 'item'][attr_names[0]] = edge_index_neutral
    data_neutral['item', 'rated_by', 'user'][attr_names[0]] = edge_index_neutral.flip([0])

    # validation
    if load_val_or_test == 'val':
        edge_label_index = torch.tensor(np.stack([df_validation['user_id:token'].values, df_validation['item_id:token'].values]))
        data['user', 'rates', 'item'][attr_names[1]] = edge_label_index

        print('user: %d,  item: %d' %(data[node_types[0]].num_nodes, data[node_types[1]].num_nodes))
        print('train interations:', len(df_train))
        print('valid interations:', len(df_validation))
        print('test interations:', len(df_test))

    # test
    elif load_val_or_test == 'test':
        edge_label_index = torch.tensor(np.stack([df_test['user_id:token'].values, df_test['item_id:token'].values]))
        data['user', 'rates', 'item'][attr_names[1]] = edge_label_index

    else:
        print('load_val_or_test error')

    return data, data_neg, data_neutral