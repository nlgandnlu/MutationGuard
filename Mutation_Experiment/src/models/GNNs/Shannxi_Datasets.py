import os
import os.path as osp
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Dataset,Data
from sklearn.preprocessing import MinMaxScaler
import gc
import time
import datetime
import pandas as pd
import math
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.stats import zscore
from tqdm import tqdm
from scipy.stats import stats
import shutil
#加载数据集，用来实现我们的方法
class Graph_Dataset2(Dataset):
    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.mode=mode

    @property
    def raw_file_names(self):
        self.raw_file_users=self.raw_dir
        self.raw_file_users_white=self.raw_dir+'/白名单用户.csv'
        self.raw_file_users_fraud=self.raw_dir+'/纯涉诈卡.csv'
        self.raw_file_users_mutation=self.raw_dir+'/突变涉诈卡.csv'
        self.raw_file_white_opusers=self.raw_dir+'/白名单对端号码.csv'
        self.raw_file_fraud_opusers=self.raw_dir+'/纯涉诈卡对端号码.csv'
        self.raw_file_change_opusers=self.raw_dir+'/突变涉诈卡对端号码.csv'
        self.raw_file_white_voc=self.raw_dir+'/白名单话单.csv'
        self.raw_file_fraud_voc=self.raw_dir+'/纯涉诈卡话单.csv'
        self.raw_file_change_voc=self.raw_dir+'/突变涉诈卡话单.csv'
        return [self.raw_file_users,self.raw_file_white_opusers,self.raw_file_fraud_opusers,self.raw_file_change_opusers,self.raw_file_white_voc,self.raw_file_fraud_voc,self.raw_file_change_voc]

    @property
    def processed_file_names(self):
        self.start_date = datetime.datetime(2023,8,1)   # 第一个日期
        self.ratio=[0.6,0.2,0.2]
        self.train_name=[]
        self.val_name=[]
        self.val_name=[]
        self.voc_fraud = pd.read_csv(self.raw_file_fraud_voc, dtype='str')
        self.voc_change = pd.read_csv(self.raw_file_change_voc, dtype='str')
        self.voc_fraud['label'] = 1
        self.voc_change['label'] = 1
        print('纯诈骗话单读取矩阵形状：')
        print(self.voc_fraud.shape)
        print('突变诈骗话单读取矩阵形状：')
        print(self.voc_change.shape)
        self.voc_non_fraud = pd.read_csv(self.raw_file_white_voc, dtype='str')
        self.voc_non_fraud['label'] = 0
        print('非诈骗话单读取矩阵形状：')
        print(self.voc_non_fraud.shape)

        idx=0
        voc_fraud_name=[]
        for x in self.voc_fraud['号码'].unique():
            voc_fraud_name.append('fraud_'+str(idx)+'.pt')
            idx+=1
        idx=0
        voc_change_name=[]
        for x in self.voc_change['号码'].unique():
            voc_change_name.append('change_'+str(idx)+'.pt')
            idx+=1
        idx=0
        voc_non_fraud_name=[]
        for x in self.voc_non_fraud['号码'].unique():
            voc_non_fraud_name.append('non_fraud_'+str(idx)+'.pt')
            idx+=1
        #file_list = os.listdir(self.processed_dir)
        all=voc_fraud_name+voc_non_fraud_name+voc_change_name
        #划分数据集
        train_fraud = int(self.ratio[0] * len(voc_fraud_name))
        val_fraud = int(self.ratio[1] * len(voc_fraud_name))
        train_change = int(self.ratio[0] * len(voc_change_name))
        val_change = int(self.ratio[1] * len(voc_change_name))
        train_non_fraud = int(self.ratio[0] * len(voc_non_fraud_name))
        val_non_fraud = int(self.ratio[1] * len(voc_non_fraud_name))
        # 设置随机种子
        random_seed = 0
        random.seed(random_seed)
        random.shuffle(voc_fraud_name)
        random.shuffle(voc_change_name)
        random.shuffle(voc_non_fraud_name)
        self.train_name=voc_fraud_name[:train_fraud]+voc_non_fraud_name[:train_non_fraud]+voc_change_name[:train_change]
        self.val_name=voc_fraud_name[train_fraud:train_fraud+val_fraud]+voc_non_fraud_name[train_non_fraud:train_non_fraud+val_non_fraud]+voc_change_name[train_change:train_change+val_change]
        self.test_name=voc_fraud_name[train_fraud+val_fraud:]+voc_non_fraud_name[train_non_fraud+val_non_fraud:]+voc_change_name[train_change+val_change:]
        self.val_change_name=voc_change_name[train_change:train_change+val_change]
        self.test_change_name=voc_change_name[train_change+val_change:]
        self.val_fraud_name=voc_fraud_name[train_fraud:train_fraud+val_fraud]
        self.test_fraud_name=voc_fraud_name[train_fraud+val_fraud:]
        print('train,val,test:',len(self.train_name),len(self.val_name),len(self.test_name))
        self.all_processed_files=all

        #self.file_out(self.test_name)

        return [self.processed_dir+'_dataset2/'+x for x in all]
        #return [self.processed_dir+'/'+'aa.pt']
    # 自定义函数：根据字符串长度提取不同位置的字符
    
    def extract_user_columns_to_tensor(self,row,required_columns):
        # 检查DataFrame中是否包含所需的列名
        missing_columns = [col for col in required_columns if col not in row.columns]

        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # 抽取所需的列值
        extracted_values = row[required_columns].values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1)

        return tensor
    def extract_voc_columns_to_tensor(self,row,required_columns):
        # 检查DataFrame中是否包含所需的列名
        missing_columns = [col for col in required_columns if col not in row.index.tolist()]

        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # 抽取所需的列值
        extracted_values = row[required_columns].values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1)

        return tensor
    def trans(self, index, attr):
        #先进行字典初始化
        dic={}
        index_new=[]
        attr_new=[]
        for edge,t in zip(index,attr):
            sub_tuple = tuple(sorted(edge))
            if sub_tuple not in dic.keys():
                dic[sub_tuple]=[]
                dic[sub_tuple].append(t)
            else:
                dic[sub_tuple].append(t)
            
        #进行边特征计算
        for edge in dic.keys():
            index_new.append(edge)
            # 计算张量的平均值
            mean_tensor = torch.stack(dic[edge]).mean(dim=0)
            # 计算总张量个数
            total_tensors = len(dic[edge])
            # 将总张量个数添加到平均张量中
            mean_tensor_with_count = torch.cat((mean_tensor, torch.tensor([[total_tensors]])), dim=1)
            attr_new.append(mean_tensor_with_count.type(torch.float32))

        # 将结果列表转换回张量
        index_new=torch.tensor(index_new)
        attr_new=torch.cat(attr_new,dim=0)
        return index_new,attr_new
    def get_obj(self,voc,user,user_op,name,prefix):
        #先进行数据分组
        grouped = voc.groupby('号码')
        data_dic={}
        idx_non_fraud=0
        idx_fraud=0
        idx_change=0
        node_dummies_columns = user.filter(like='dummies', axis=1)
        node_required_columns = ['年龄','本月消费','本月使用流量/M','本月通话时长','3个月主叫占比','3个月主叫离散','3个月主叫外省号码占比']
        node_required_columns.extend(node_dummies_columns)
        voc_dummies_columns = voc.filter(like='dummies', axis=1)
        voc_required_columns = ['x','y','z','通话时长']
        voc_required_columns.extend(voc_dummies_columns)
        for target_number in tqdm(name, desc="Processing"):
            group_a = grouped.get_group(target_number)  # 提取名为target_number的分组数据
            #存储号码序号的映射字典
            dic={target_number:0}
            nodes=[]
            edges=[]
            voc_tensors=[]
            poss=[]
            label='0'
            #设置是否为根节点的标识
            root=[]
            #先添加根节点
            #可能存在查不到的情况，需要设置为0
            if target_number in user['号码'].values:
                node_tensor=self.extract_user_columns_to_tensor(user[user['号码'] == target_number],node_required_columns)
            else:
                node_tensor=torch.zeros(len(node_required_columns))
            nodes.append(node_tensor.type(torch.float32).reshape(-1,node_tensor.shape[-1]))
            root.append(1)
            for index, row in group_a.iterrows():
                #对上述列表逐个添加特征
                if row['对端号码'] not in dic.keys():
                    dic[row['对端号码']]=len(dic.keys())
                    #可能存在查不到的情况，需要设置为0
                    if row['对端号码'] in user_op['号码'].values:
                        node_tensor=self.extract_user_columns_to_tensor(user_op[user_op['号码'] == row['对端号码']],node_required_columns)
                    else:
                        node_tensor=torch.zeros(len(node_required_columns)).type(torch.float32)
                    
                    nodes.append(node_tensor.reshape(-1,node_tensor.shape[-1]).type(torch.float32))
                    root.append(0)
                edges.append([dic[target_number],dic[row['对端号码']]])
                voc_tensor = self.extract_voc_columns_to_tensor(row,voc_required_columns).type(torch.float32)
                voc_tensors.append(voc_tensor[0,3:].reshape(-1,voc_tensor[0,3:].shape[-1]))
                poss.append(voc_tensor[0,:3].reshape(-1,3))
                label=row['label']
            #统计获取边特征，并且去掉重复边
            edges,edge_tensors=self.trans(edges,voc_tensors)
            nodes=torch.cat(nodes,dim=0)
            voc_tensors=torch.cat(voc_tensors,dim=0)
            poss=torch.cat(poss,dim=0)
            label=torch.tensor(label)
            root=torch.tensor(root)
            #从数据中创建一个DATA对象
            data=Data(x=nodes,edge_index=edges,edge_attr=edge_tensors,voc_attr=voc_tensors,y=label,pos=poss,root=root)
            if idx_fraud==0:
                print(data)
            if prefix=='non_fraud_':
                data_dic['non_fraud_'+str(idx_non_fraud)+'.pt']=data
                idx_non_fraud+=1
            elif prefix=='fraud_':
                data_dic['fraud_'+str(idx_fraud)+'.pt']=data
                idx_fraud+=1
            elif prefix=='change_':
                data_dic['change_'+str(idx_change)+'.pt']=data
                idx_change+=1
        return data_dic
    def get_users(self,data_path):
        user_fraud = pd.read_csv(self.raw_file_users_fraud, dtype='str')
        user_change = pd.read_csv(self.raw_file_users_mutation, dtype='str')
        user_non_fraud = pd.read_csv(self.raw_file_users_white, dtype='str')
        user_fraud_opusers = pd.read_csv(self.raw_file_fraud_opusers, dtype='str')
        user_change_opusers = pd.read_csv(self.raw_file_change_opusers, dtype='str')
        user_non_fraud_opusers = pd.read_csv(self.raw_file_white_opusers, dtype='str')
        all=[user_fraud , user_change, user_non_fraud,user_fraud_opusers,user_change_opusers,user_non_fraud_opusers]
        length=[len(x) for x in all]
        for i in range(1,len(length)):
            length[i]=length[i]+length[i-1]
        # 在行方向上拼接数据
        df = pd.concat(all, ignore_index=True)
        df=self.get_user_embed(df)
        return df[:length[0]],df[length[0]:length[1]],df[length[1]:length[2]],df[length[2]:length[3]],df[length[3]:length[4]],df[length[4]:length[5]],df[:length[2]]
    def get_user_embed(self,original_df):
        encoded_df = pd.get_dummies(original_df, columns=['开户接入方式','宽带标识'], prefix=['开户接入方式_dummies','宽带标识_dummies'])
        original_df=original_df[['开户接入方式','宽带标识']].join(encoded_df)
        z_list=['年龄','本月消费','本月使用流量/M','本月通话时长','3个月主叫占比','3个月主叫离散','3个月主叫外省号码占比']
        for name in z_list:
            original_df[name].fillna(0, inplace=True)
        return original_df
    def extract_hours(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[:2]
    # 自定义函数：根据字符串长度提取不同位置的字符
    def extract_minutes(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[2:4]
    # 自定义函数：根据字符串长度提取不同位置的字符
    def extract_seconds(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[4:]
    def cylindrical_to_cartesian(self, r, theta, h):
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = h
        return x, y, z
    def cacu_time(self,df):
        z = (datetime.datetime(int(df['year']), int(df['month']), int(df['day']))-self.start_date).days
        r = 1
        alltime=24*60*60
        now=int(df['hour'])*3600+int(df['minute'])*60+int(df['second'])
        theta = now/alltime*2*math.pi
        x, y, z = self.cylindrical_to_cartesian(r, theta, z)
        return x,y,z
    # 获取通话时间特征
    def get_voc_feat(self,df):
        print('正在处理时间列。。。')
        df["start_datetime"] = pd.to_datetime(df['日期'])
        df["year"] = df["start_datetime"].dt.year
        df["month"] = df["start_datetime"].dt.month
        df["day"] = df["start_datetime"].dt.day
        df['hour'] = df['时间'].apply(self.extract_hours).astype('int64')
        df['minute'] = df['时间'].apply(self.extract_minutes).astype('int64')
        df['second'] = df['时间'].apply(self.extract_seconds).astype('int64')
        print('--------------已添加时间列！-----------')
        return df
    # 读取原始数据，对voc数据进行编码
    def get_voc_embed(self,original_df):
        #计算地址变化情况,注意需要确保话单是按照时间顺序的
        # 初始化新的列
        original_df['漫游地和对端号码归属相同'] = '-1'
        original_df['imei变化'] = '-1'
        original_df['小区变化'] = '-1'
        original_df['基站变化'] = '-1'
        original_df['漫游地变化'] = '-1'
        original_df['x'] = '0'
        original_df['y'] = '0'
        original_df['z'] = '0'
        # 遍历数据并更新列
        print('正在遍历voc数据，并进行处理。。。')
        for i, row in tqdm(original_df.iterrows(), total=len(original_df)):
            # 对每行额外更新通话时长和坐标信息
            if original_df.at[i, '呼叫类型'] == '2':
                original_df.at[i, '通话时长'] = -int(original_df.at[i, '通话时长'])
            else:
                original_df.at[i, '通话时长'] = int(original_df.at[i, '通话时长'])
            original_df.at[i, 'x'],original_df.at[i, 'y'],original_df.at[i, 'z']=self.cacu_time(original_df.iloc[i])
            # 判断漫游地和对端号码归属是否相同,先保证值不为空
            if pd.isna(original_df.at[i, '漫游地']) or pd.isna(original_df.at[i, '对端号码归属']):
                pass
            elif original_df.at[i, '漫游地'] == original_df.at[i, '对端号码归属']:
                original_df.at[i, '漫游地和对端号码归属相同'] = '1'
            #必须要求是同一个电话号码才开始比较
            if i>0 and original_df.at[i, '号码'] == original_df.at[i - 1, '号码']:
                # 判断属性是否变化
                if pd.isna(original_df.at[i, 'imei']) or pd.isna(original_df.at[i - 1, 'imei']):
                    pass
                elif original_df.at[i, 'imei'] != original_df.at[i - 1, 'imei']:
                    original_df.at[i, 'imei变化'] = '1'
                if pd.isna(original_df.at[i, '小区']) or pd.isna(original_df.at[i - 1, '小区']):
                    pass
                elif original_df.at[i, '小区'] != original_df.at[i - 1, '小区']:
                    original_df.at[i, '小区变化'] = '1'
                if pd.isna(original_df.at[i, '基站']) or pd.isna(original_df.at[i - 1, '基站']):
                    pass
                elif original_df.at[i, '基站'] != original_df.at[i - 1, '基站']:
                    original_df.at[i, '基站变化'] = '1'
                if pd.isna(original_df.at[i, '漫游地']) or pd.isna(original_df.at[i - 1, '漫游地']):
                    pass
                elif original_df.at[i, '漫游地'] != original_df.at[i - 1, '漫游地']:
                    original_df.at[i, '漫游地变化'] = '1'

        #对给定的属性列获取独热编码并存储在'dummies'列
        encoded_df = pd.get_dummies(original_df, columns=['呼叫类型','漫游地和对端号码归属相同','imei变化','小区变化','基站变化','漫游地变化'], prefix=['呼叫类型_dummies','漫游地和对端号码归属相同_dummies','imei变化_dummies','小区变化_dummies','基站变化_dummies','漫游地变化_dummies'])
        original_df=original_df[['呼叫类型','漫游地和对端号码归属相同','imei变化','小区变化','基站变化','漫游地变化']].join(encoded_df)
        z_list=['通话时长']
        for name in z_list:
            original_df[name].fillna(0, inplace=True)
        height=2
        original_df['z'] = (original_df['z'] - original_df['z'].min()) / (original_df['z'].max() - original_df['z'].min()) * height
        return original_df
    def get_and_transfer(self,data_path):
        len_fraud=self.voc_fraud.shape[0]
        len_change=self.voc_change.shape[0]
        len_non_fraud=self.voc_non_fraud.shape[0]
        # 在行方向上拼接数据
        df = pd.concat([self.voc_fraud, self.voc_change, self.voc_non_fraud], ignore_index=True)
        # 获取通话时间信息 这步骤就是把通话的时间拆分成了小时，分钟等等单独的信息保存在了通话信息表中
        df = self.get_voc_feat(df)
        #对df直接获取编码
        df=self.get_voc_embed(df)
        return df[:len_fraud],df[len_fraud:len_fraud+len_change],df[len_fraud+len_change:],df
    def get_data_obj(self,data_path):
        #目的是获取带有label的voc信息，把内部数据集的voc数据格式转化为公开数据集格式(额外多一行label信息)，然后直接复用代码
        voc_fraud,voc_change,voc_non_fraud,voc_ori=self.get_and_transfer(data_path)
        #分别获得用户信息，并进行编码
        user_fraud,user_change,user_non_fraud,user_fraud_op,user_change_op,user_non_fraud_op,user_ori=self.get_users(data_path)
        fraud_name=voc_fraud['号码'].unique()
        change_name=voc_change['号码'].unique()
        non_fraud_name=voc_non_fraud['号码'].unique()

        #对通话voc记录进行保存，保存成图需要的Data对象
        all={}
        print('正在处理graph信息获得obj对象')
        fraud_obj=self.get_obj(voc_fraud,user_fraud,user_fraud_op,fraud_name,'fraud_')
        change_obj=self.get_obj(voc_change,user_change,user_change_op,change_name,'change_')
        #dic obj,需要用名字作为key以便后续存储
        non_fraud_obj=self.get_obj(voc_non_fraud,user_non_fraud,user_non_fraud_op,non_fraud_name,'non_fraud_')
        all.update(fraud_obj)
        all.update(change_obj)
        all.update(non_fraud_obj)
        return fraud_obj,change_obj,non_fraud_obj,all
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    def process(self):
        idx = 0
        exist_files=os.listdir(self.processed_dir+'_dataset2/')
        # 检查列表a是否全部包含在file_names中
        all_included = all(file_name in exist_files for file_name in self.all_processed_files)
        if not all_included:
            fraud_obj,change_obj,non_fraud_obj,all_obj=self.get_data_obj(self.raw_file_users)
            print('共有'+str(len(fraud_obj))+'个诈骗对象，正在输出')
            print('共有'+str(len(change_obj))+'个突变诈骗对象，正在输出')
            print('共有'+str(len(non_fraud_obj))+'个正常对象，正在输出')
            for name in all_obj.keys():
                data = all_obj[name]

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir+'_dataset2/', name))
                idx += 1
        else:
            print('文件已经都处理好，无需进一步处理')
    def change_mode(self,mode):
        self.mode=mode
    def len(self):
        if self.mode=='train':
            return len(self.train_name)
        elif self.mode=='val':
            return len(self.val_name)
        elif self.mode=='test':
            return len(self.test_name)
        elif self.mode=='change':
            return len(self.test_change_name)
        elif self.mode=='fraud':
            return len(self.test_fraud_name)
    def get_tranformer(self):
        name_list=self.train_name
        all_x=[]
        all_y=[]
        for n in name_list:
            data = torch.load(osp.join(self.processed_dir+'_dataset2/', n))
            root_index= torch.nonzero(data.root).squeeze()
            feature=data.x[root_index].reshape(-1,data.x.shape[-1]).type(torch.float32)
            label=data.y
            all_x.append(feature)
            all_y.append(label)
        x=torch.cat(all_x,dim=0).cpu().numpy()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit_transform(x)
    def get(self, idx):
        #需要进行一些数据的维度翻转（适应类定义）和存储名字更改（dataloader的batch会根据x识别，所以不能直接存储）
        if self.mode=='train':
            data = torch.load(osp.join(self.processed_dir+'_dataset2/', self.train_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='val':
            data = torch.load(osp.join(self.processed_dir+'_dataset2/', self.val_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='test':
            data = torch.load(osp.join(self.processed_dir+'_dataset2/', self.test_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='change':
            data = torch.load(osp.join(self.processed_dir+'_dataset2/', self.test_change_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='fraud':
            data = torch.load(osp.join(self.processed_dir+'_dataset2/', self.test_fraud_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.edge_index=data.edge_index.permute(1, 0)
        
        return data
    
#加载数据集，通话记录信息由统计获得，不附带边特征
class Graph_Dataset3(Dataset):
    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.mode=mode

    @property
    def raw_file_names(self):
        self.raw_file_users=self.raw_dir
        self.raw_file_users_white=self.raw_dir+'/白名单用户.csv'
        self.raw_file_users_fraud=self.raw_dir+'/纯涉诈卡.csv'
        self.raw_file_users_mutation=self.raw_dir+'/突变涉诈卡.csv'
        self.raw_file_white_opusers=self.raw_dir+'/白名单对端号码.csv'
        self.raw_file_fraud_opusers=self.raw_dir+'/纯涉诈卡对端号码.csv'
        self.raw_file_change_opusers=self.raw_dir+'/突变涉诈卡对端号码.csv'
        self.raw_file_white_voc=self.raw_dir+'/白名单话单.csv'
        self.raw_file_fraud_voc=self.raw_dir+'/纯涉诈卡话单.csv'
        self.raw_file_change_voc=self.raw_dir+'/突变涉诈卡话单.csv'
        return [self.raw_file_users,self.raw_file_white_opusers,self.raw_file_fraud_opusers,self.raw_file_change_opusers,self.raw_file_white_voc,self.raw_file_fraud_voc,self.raw_file_change_voc]

    @property
    def processed_file_names(self):
        self.start_date = datetime.datetime(2023,8,1)   # 第一个日期
        self.ratio=[0.6,0.2,0.2]
        self.train_name=[]
        self.val_name=[]
        self.val_name=[]
        self.voc_fraud = pd.read_csv(self.raw_file_fraud_voc, dtype='str')
        self.voc_change = pd.read_csv(self.raw_file_change_voc, dtype='str')
        self.voc_fraud['label'] = 1
        self.voc_change['label'] = 1
        print('纯诈骗话单读取矩阵形状：')
        print(self.voc_fraud.shape)
        print('突变诈骗话单读取矩阵形状：')
        print(self.voc_change.shape)
        self.voc_non_fraud = pd.read_csv(self.raw_file_white_voc, dtype='str')
        self.voc_non_fraud['label'] = 0
        print('非诈骗话单读取矩阵形状：')
        print(self.voc_non_fraud.shape)

        idx=0
        voc_fraud_name=[]
        for x in self.voc_fraud['号码'].unique():
            voc_fraud_name.append('fraud_'+str(idx)+'.pt')
            idx+=1
        idx=0
        voc_change_name=[]
        for x in self.voc_change['号码'].unique():
            voc_change_name.append('change_'+str(idx)+'.pt')
            idx+=1
        idx=0
        voc_non_fraud_name=[]
        for x in self.voc_non_fraud['号码'].unique():
            voc_non_fraud_name.append('non_fraud_'+str(idx)+'.pt')
            idx+=1
        all=voc_fraud_name+voc_non_fraud_name+voc_change_name
        #划分数据集
        train_fraud = int(self.ratio[0] * len(voc_fraud_name))
        val_fraud = int(self.ratio[1] * len(voc_fraud_name))
        train_change = int(self.ratio[0] * len(voc_change_name))
        val_change = int(self.ratio[1] * len(voc_change_name))
        train_non_fraud = int(self.ratio[0] * len(voc_non_fraud_name))
        val_non_fraud = int(self.ratio[1] * len(voc_non_fraud_name))
        # 设置随机种子
        random_seed = 0
        random.seed(random_seed)
        random.shuffle(voc_fraud_name)
        random.shuffle(voc_change_name)
        random.shuffle(voc_non_fraud_name)
        self.train_name=voc_fraud_name[:train_fraud]+voc_non_fraud_name[:train_non_fraud]+voc_change_name[:train_change]
        self.val_name=voc_fraud_name[train_fraud:train_fraud+val_fraud]+voc_non_fraud_name[train_non_fraud:train_non_fraud+val_non_fraud]+voc_change_name[train_change:train_change+val_change]
        self.test_name=voc_fraud_name[train_fraud+val_fraud:]+voc_non_fraud_name[train_non_fraud+val_non_fraud:]+voc_change_name[train_change+val_change:]
        self.val_change_name=voc_change_name[train_change:train_change+val_change]
        self.test_change_name=voc_change_name[train_change+val_change:]
        self.val_fraud_name=voc_fraud_name[train_fraud:train_fraud+val_fraud]
        self.test_fraud_name=voc_fraud_name[train_fraud+val_fraud:]
        print('train,val,test:',len(self.train_name),len(self.val_name),len(self.test_name))
        self.all_processed_files=all
        return [self.processed_dir+'_dataset3/'+x for x in all]
    # 自定义函数：根据字符串长度提取不同位置的字符
    
    def extract_user_columns_to_tensor(self,row,required_columns):
        # 检查DataFrame中是否包含所需的列名
        missing_columns = [col for col in required_columns if col not in row.columns]

        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # 抽取所需的列值
        extracted_values = row[required_columns].values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1)

        return tensor
    def extract_voc_columns_to_tensor(self,row):
        # 抽取所需的列值
        row.drop(columns=['号码'], inplace=True)
        extracted_values = row.values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1)

        return tensor
    def trans(self, index):
        #先进行字典初始化
        dic={}
        index_new=[]
        for edge in index:
            sub_tuple = tuple(sorted(edge))
            if sub_tuple not in dic.keys():
                dic[sub_tuple]=[]
            else:
                pass
            
        for edge in dic.keys():
            index_new.append(edge)

        # 将结果列表转换回张量
        index_new=torch.tensor(index_new)
        return index_new
    def get_obj(self,voc,user,user_op,name,prefix,voc_new):
        #先进行数据分组
        grouped = voc.groupby('号码')
        data_dic={}
        idx_non_fraud=0
        idx_fraud=0
        idx_change=0
        node_dummies_columns = user.filter(like='dummies', axis=1)
        node_required_columns = ['年龄','本月消费','本月使用流量/M','本月通话时长','3个月主叫占比','3个月主叫离散','3个月主叫外省号码占比']
        node_required_columns.extend(node_dummies_columns)
        for target_number in tqdm(name, desc="Processing"):
            group_a = grouped.get_group(target_number)  # 提取名为target_number的分组数据
            #存储号码序号的映射字典
            dic={target_number:0}
            nodes=[]
            edges=[]
            label='0'
            #设置是否为根节点的标识
            root=[]
            #先添加根节点
            #可能存在查不到的情况，需要设置为0
            if target_number in user['号码'].values:
                node_tensor=self.extract_user_columns_to_tensor(user[user['号码'] == target_number],node_required_columns)
            else:
                node_tensor=torch.zeros(len(node_required_columns))
            voc_tensor=self.extract_voc_columns_to_tensor(voc_new[voc_new['号码'] == target_number])
            node_tensor=torch.cat([voc_tensor,node_tensor.reshape(-1,node_tensor.shape[-1])],dim=-1)
            nodes.append(node_tensor.type(torch.float32).reshape(-1,node_tensor.shape[-1]))
            root.append(1)
            for index, row in group_a.iterrows():
                #对上述列表逐个添加特征
                if row['对端号码'] not in dic.keys():
                    dic[row['对端号码']]=len(dic.keys())
                    #可能存在查不到的情况，需要设置为0
                    if row['对端号码'] in user_op['号码'].values:
                        node_tensor=self.extract_user_columns_to_tensor(user_op[user_op['号码'] == row['对端号码']],node_required_columns)
                    else:
                        node_tensor=torch.zeros(len(node_required_columns)).type(torch.float32)
                    node_tensor=torch.cat([torch.zeros_like(voc_tensor),node_tensor.reshape(-1,node_tensor.shape[-1])],dim=-1)
                    nodes.append(node_tensor.reshape(-1,node_tensor.shape[-1]).type(torch.float32))
                    root.append(0)
                edges.append([dic[target_number],dic[row['对端号码']]])
                label=row['label']
            #统计获取边特征，并且去掉重复边
            edges=self.trans(edges)
            nodes=torch.cat(nodes,dim=0)
            label=torch.tensor(label)
            root=torch.tensor(root)
            #从数据中创建一个DATA对象
            data=Data(x=nodes,edge_index=edges,y=label,root=root)
            if prefix=='non_fraud_':
                data_dic['non_fraud_'+str(idx_non_fraud)+'.pt']=data
                idx_non_fraud+=1
            elif prefix=='fraud_':
                data_dic['fraud_'+str(idx_fraud)+'.pt']=data
                idx_fraud+=1
            elif prefix=='change_':
                data_dic['change_'+str(idx_change)+'.pt']=data
                idx_change+=1
        return data_dic
    def get_users(self,data_path):
        user_fraud = pd.read_csv(self.raw_file_users_fraud, dtype='str')
        user_change = pd.read_csv(self.raw_file_users_mutation, dtype='str')
        user_non_fraud = pd.read_csv(self.raw_file_users_white, dtype='str')
        user_fraud_opusers = pd.read_csv(self.raw_file_fraud_opusers, dtype='str')
        user_change_opusers = pd.read_csv(self.raw_file_change_opusers, dtype='str')
        user_non_fraud_opusers = pd.read_csv(self.raw_file_white_opusers, dtype='str')
        all=[user_fraud , user_change, user_non_fraud,user_fraud_opusers,user_change_opusers,user_non_fraud_opusers]
        length=[len(x) for x in all]
        for i in range(1,len(length)):
            length[i]=length[i]+length[i-1]
        # 在行方向上拼接数据
        df = pd.concat(all, ignore_index=True)
        df=self.get_user_embed(df)
        return df[:length[0]],df[length[0]:length[1]],df[length[1]:length[2]],df[length[2]:length[3]],df[length[3]:length[4]],df[length[4]:length[5]],df[:length[2]]
    def get_user_embed(self,original_df):
        #对给定的属性列获取独热编码并存储在'hot'
        encoded_df = pd.get_dummies(original_df, columns=['开户接入方式','宽带标识'], prefix="dummies")
        original_df=original_df[['开户接入方式','宽带标识']].join(encoded_df)
        z_list=['年龄','本月消费','本月使用流量/M','本月通话时长','3个月主叫占比','3个月主叫离散','3个月主叫外省号码占比']
        for name in z_list:
            original_df[name].fillna(0, inplace=True)
        return original_df
    def extract_hours(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[:2]
    # 自定义函数：根据字符串长度提取不同位置的字符
    def extract_minutes(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[2:4]
    # 自定义函数：根据字符串长度提取不同位置的字符
    def extract_seconds(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[4:]
    def cylindrical_to_cartesian(self, r, theta, h):
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = h
        return x, y, z
    def cacu_time(self,df):
        z = (datetime.datetime(int(df['year']), int(df['month']), int(df['day']))-self.start_date).days
        r = 1
        alltime=24*60*60
        now=int(df['hour'])*3600+int(df['minute'])*60+int(df['second'])
        theta = now/alltime*2*math.pi
        x, y, z = self.cylindrical_to_cartesian(r, theta, z)
        return x,y,z
    def get_voc_feat(self, df):
        print('开始处理通话数据')
        df["start_datetime"] = pd.to_datetime(df['日期'])
        df["hour"] = df['时间'].apply(self.extract_hours).astype('int64')
        df["day"] = df['start_datetime'].dt.day
        df["通话时长"] = df['通话时长'].astype('int64')
        phone_no_m = df[["号码"]].copy()
        #phone_no_m = phone_no_m.drop_duplicates(subset=['phone_no_m'], keep='last')：去除了重复的 phone_no_m 值，只保留最后出现的。
        phone_no_m = phone_no_m.drop_duplicates(subset=['号码'], keep='last')
        # 对话人数和对话次数
        print('     正在计算通话次数和通话人数')
        tmp = df.groupby("号码")["对端号码"].agg(opposite_count="count", opposite_unique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")
        """主叫通话
        """
        print('正在处理通话类型为主叫的电话信息：')
        print('     正在计算imeis个数')
        df_call = df[df["呼叫类型"] == '1'].copy()
        tmp = df_call.groupby("号码")["imei"].agg(voccalltype1="count", imeis="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")
        print('     正在计算主叫占比')
        phone_no_m["voc_calltype1"] = phone_no_m["voccalltype1"] / phone_no_m["opposite_count"]
        print('     正在计算通话类型个数')
        tmp = df.groupby("号码")["呼叫类型"].agg(calltype_id_unique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")

        """和固定通话者的对话统计
        """
        print('正在统计通话交互行为信息:')

        tmp = df.groupby(["号码", "对端号码"])["通话时长"].agg(count="count", sum="sum")
        print('     正在统计和固定通话者的通话次数信息')
        phone2opposite = tmp.groupby("号码")["count"].agg(phone2opposite_mean="mean"
                                                                , phone2opposite_median="median"
                                                                , phone2opposite_max="max"
                                                                , phone2opposite_min="min"
                                                                , phone2opposite_var="var"
                                                                , phone2opposite_skew="skew"
                                                                , phone2opposite_sem="sem"
                                                                , phone2opposite_std="std"
                                                                , phone2opposite_quantile="quantile"
                                                                )

        phone_no_m = phone_no_m.merge(phone2opposite, on="号码", how="left")
        print('     正在统计和固定通话者的通话总时长信息')
        phone2opposite = tmp.groupby("号码")["sum"].agg(phone2oppo_sum_mean="mean"
                                                            , phone2oppo_sum_median="median"
                                                            , phone2oppo_sum_max="max"
                                                            , phone2oppo_sum_min="min"
                                                            , phone2oppo_sum_var="var"
                                                            , phone2oppo_sum_skew="skew"
                                                            , phone2oppo_sum_sem="sem"
                                                            , phone2oppo_sum_std="std"
                                                            , phone2oppo_sum_quantile="quantile"
                                                            )

        phone_no_m = phone_no_m.merge(phone2opposite, on="号码", how="left")

        """通话时间长短统计
        """
        print('     正在统计和固定通话者的每次通话时长信息')
        tmp = df.groupby("号码")["通话时长"].agg(call_dur_mean="mean"
                                                    , call_dur_median="median"
                                                    , call_dur_max="max"
                                                    , call_dur_min="min"
                                                    , call_dur_var="var"
                                                    , call_dur_skew="skew"
                                                    , call_dur_sem="sem"
                                                    , call_dur_std="std"
                                                    , call_dur_quantile="quantile"
                                                    )
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")

        tmp = df.groupby("号码")["对端号码归属"].agg(city_name_nunique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")
        tmp = df.groupby("号码")["呼叫类型"].agg(calltype_id_unique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")

        """通话时间点偏好
        """
        print('正在处理通话时间偏好信息：')
        print('     正在计算每日最常通话时间点，及在该时间点通话次数，通话时间分布')
        tmp = df.groupby("号码")["hour"].agg(voc_hour_mode=lambda x: stats.mode(x)[0][0],
                                                voc_hour_mode_count=lambda x: stats.mode(x)[1][0],
                                                voc_hour_nunique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")
        print('     正在计算每月最常通话日期，及在该日期通话次数，通话时间分布')
        tmp = df.groupby("号码")["day"].agg(voc_day_mode=lambda x: stats.mode(x)[0][0],
                                                voc_day_mode_count=lambda x: stats.mode(x)[1][0],
                                                voc_day_nunique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")

        phone_no_m.fillna(0, inplace=True)
        z_list = phone_no_m.select_dtypes(include=['number'])
        for name in z_list:
            phone_no_m[name].fillna(0, inplace=True)
        return phone_no_m
    def get_and_transfer(self,data_path):
        return self.get_voc_feat(self.voc_fraud),self.get_voc_feat(self.voc_change),self.get_voc_feat(self.voc_non_fraud)
    def get_data_obj(self,data_path):
        #目的是获取带有label的voc信息
        voc_fraud_new,voc_change_new,voc_non_fraud_new=self.get_and_transfer(data_path)
        
        #分别获得用户信息，并进行编码
        user_fraud,user_change,user_non_fraud,user_fraud_op,user_change_op,user_non_fraud_op,user_ori=self.get_users(data_path)
        
        fraud_name=voc_fraud_new['号码'].unique()
        change_name=voc_change_new['号码'].unique()
        non_fraud_name=voc_non_fraud_new['号码'].unique()

        #对通话voc记录进行保存，保存成图需要的Data对象
        all={}
        print('正在处理graph信息获得obj对象')
        fraud_obj=self.get_obj(self.voc_fraud,user_fraud,user_fraud_op,fraud_name,'fraud_',voc_fraud_new)
        change_obj=self.get_obj(self.voc_change,user_change,user_change_op,change_name,'change_',voc_change_new)
        #dic obj,需要用名字作为key以便后续存储
        non_fraud_obj=self.get_obj(self.voc_non_fraud,user_non_fraud,user_non_fraud_op,non_fraud_name,'non_fraud_',voc_non_fraud_new)
        all.update(fraud_obj)
        all.update(change_obj)
        all.update(non_fraud_obj)
        return fraud_obj,change_obj,non_fraud_obj,all
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    def process(self):
        idx = 0
        exist_files=os.listdir(self.processed_dir+'_dataset3/')
        # 检查列表a是否全部包含在file_names中
        all_included = all(file_name in exist_files for file_name in self.all_processed_files)
        if not all_included:
            fraud_obj,change_obj,non_fraud_obj,all_obj=self.get_data_obj(self.raw_file_users)
            print('共有'+str(len(fraud_obj))+'个诈骗对象，正在输出')
            print('共有'+str(len(change_obj))+'个突变诈骗对象，正在输出')
            print('共有'+str(len(non_fraud_obj))+'个正常对象，正在输出')
            for name in all_obj.keys():
                data = all_obj[name]

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir+'_dataset3/', name))
                idx += 1
        else:
            print('文件已经都处理好，无需进一步处理')
    def change_mode(self,mode):
        self.mode=mode
    def len(self):
        if self.mode=='train':
            return len(self.train_name)
        elif self.mode=='val':
            return len(self.val_name)
        elif self.mode=='test':
            return len(self.test_name)
        elif self.mode=='change':
            return len(self.test_change_name)
        elif self.mode=='fraud':
            return len(self.test_fraud_name)
    def get_tranformer(self):
        name_list=self.train_name
        all_x=[]
        all_y=[]
        for n in name_list:
            data = torch.load(osp.join(self.processed_dir+'_dataset3/', n))
            root_index= torch.nonzero(data.root).squeeze()
            feature=data.x[root_index].reshape(-1,data.x.shape[-1]).type(torch.float32)
            label=data.y
            all_x.append(feature)
            all_y.append(label)
        x=torch.cat(all_x,dim=0).cpu().numpy()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit_transform(x)
    def get(self, idx):
        #需要进行一些数据的维度翻转（适应类定义）和存储名字更改（dataloader的batch会根据x识别，所以不能直接存储）
        if self.mode=='train':
            data = torch.load(osp.join(self.processed_dir+'_dataset3/', self.train_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='val':
            data = torch.load(osp.join(self.processed_dir+'_dataset3/', self.val_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='test':
            data = torch.load(osp.join(self.processed_dir+'_dataset3/', self.test_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='change':
            data = torch.load(osp.join(self.processed_dir+'_dataset3/', self.test_change_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='fraud':
            data = torch.load(osp.join(self.processed_dir+'_dataset3/', self.test_fraud_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)
        
        return data

class Graph_Dataset6(Dataset):
    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.mode=mode

    @property
    def raw_file_names(self):
        self.raw_file_users=self.raw_dir+'/open_users.csv'
        self.raw_file_voc=self.raw_dir+'/open_voc.csv'
        self.train_name_01=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.1name_train.txt')]
        self.test_name_01=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.1name_test.txt')]
        self.val_name_01=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.1name_val.txt')]
        self.train_name_02=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.2name_train.txt')]
        self.test_name_02=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.2name_test.txt')]
        self.val_name_02=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.2name_val.txt')]
        self.train_name_04=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.4name_train.txt')]
        self.test_name_04=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.4name_test.txt')]
        self.val_name_04=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.4name_val.txt')]
        self.train_name_06=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.6name_train.txt')]
        self.test_name_06=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.6name_test.txt')]
        self.val_name_06=[x+'.txt' for x in self.file_in(self.raw_dir+'/0.6name_val.txt')]
        return [self.raw_file_users,self.raw_file_voc]

    def file_in(self,file_path):

        lines_list = []  # 创建一个空列表来存储读取的行

        # 打开文件并逐行读取内容
        with open(file_path, 'r') as file:
            for line in file:
                lines_list.append(line.strip())  # 将读取的每行文本添加到列表中（.strip()用于去除换行符等空白字符）
        return lines_list
    @property
    def processed_file_names(self):
        self.start_date = datetime.datetime(2019,8,1)   # 第一个日期

        self.voc = pd.read_csv(self.raw_file_voc, dtype='str')
        
        print('话单读取矩阵形状：')
        print(self.voc.shape)

        all=[]
        all.extend(self.train_name_01)
        all.extend(self.test_name_01)
        all.extend(self.val_name_01)
        self.all_processed_files=all
        self.name=[x.rstrip('.txt') for x in self.all_processed_files]
        return [self.processed_dir+'_dataset6/'+x for x in all]
    # 自定义函数：根据字符串长度提取不同位置的字符
    
    def extract_user_columns_to_tensor(self,row,required_columns):
        # 检查DataFrame中是否包含所需的列名
        missing_columns = [col for col in required_columns if col not in row.columns]

        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # 抽取所需的列值
        extracted_values = row[required_columns].values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1),row['label'].values.astype(int)

        return tensor
    def extract_voc_columns_to_tensor(self,row,required_columns):
        # 检查DataFrame中是否包含所需的列名
        missing_columns = [col for col in required_columns if col not in row.index.tolist()]

        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # 抽取所需的列值
        extracted_values = row[required_columns].values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1)

        return tensor
    def trans(self, index, attr):
        #先进行字典初始化
        dic={}
        index_new=[]
        attr_new=[]
        for edge,t in zip(index,attr):
            sub_tuple = tuple(sorted(edge))
            if sub_tuple not in dic.keys():
                dic[sub_tuple]=[]
                dic[sub_tuple].append(t)
            else:
                dic[sub_tuple].append(t)
            
        #进行边特征计算
        for edge in dic.keys():
            index_new.append(edge)
            # 计算张量的平均值
            mean_tensor = torch.stack(dic[edge]).mean(dim=0)
            # 计算总张量个数
            total_tensors = len(dic[edge])
            # 将总张量个数添加到平均张量中
            mean_tensor_with_count = torch.cat((mean_tensor, torch.tensor([[total_tensors]])), dim=1)
            attr_new.append(mean_tensor_with_count.type(torch.float32))

        # 将结果列表转换回张量
        index_new=torch.tensor(index_new)
        attr_new=torch.cat(attr_new,dim=0)
        return index_new,attr_new
    def get_obj(self,voc,user):
        #先进行数据分组
        user_op=user
        name=self.name										                                                                                                             								                                                                                  										
        grouped = voc.groupby('phone_no_m')
        data_dic={}
        idx_non_fraud=0
        idx_fraud=0

        node_required_columns = ['sms_count','sms_nunique','sms_rate','calltype_2','calltype_rate','hour_mode','hour_mode_count','hour_nunique','day_mode','day_mode_count','day_nunique','busi_count','flow_mean','flow_median','flow_min','flow_max','flow_var','flow_sum','month_ids','flow_month','arpu_mean','arpu_var','arpu_max','arpu_min','arpu_median','arpu_sum','arpu_skew','arpu_sem','arpu_quantile','opposite_count','opposite_unique','voccalltype1','imeis','voc_calltype1','calltype_id_unique_x','phone2opposite_mean','phone2opposite_median','phone2opposite_max','phone2opposite_min','phone2opposite_var','phone2opposite_skew','phone2opposite_sem','phone2opposite_std','phone2opposite_quantile','phone2oppo_sum_mean','phone2oppo_sum_median','phone2oppo_sum_max','phone2oppo_sum_min','phone2oppo_sum_var','phone2oppo_sum_skew','phone2oppo_sum_sem','phone2oppo_sum_std','phone2oppo_sum_quantile','call_dur_mean','call_dur_median','call_dur_max','call_dur_min','call_dur_var','call_dur_skew','call_dur_sem','call_dur_std','call_dur_quantile','city_name_nunique','county_name_nunique','calltype_id_unique_y','voc_hour_mode','voc_hour_mode_count','voc_hour_nunique','voc_day_mode','voc_day_mode_count','voc_day_nunique']
        voc_dummies_columns = voc.filter(like='dummies', axis=1)
        voc_required_columns = ['x','y','z','call_dur']
        voc_required_columns.extend(voc_dummies_columns)
        for target_number in tqdm(name, desc="Processing"):
            # if target_number not in voc['phone_no_m'].values:
            #     continue
            group_a = grouped.get_group(target_number)  # 提取名为target_number的分组数据
            #存储号码序号的映射字典
            dic={target_number:0}
            nodes=[]
            edges=[]
            voc_tensors=[]
            poss=[]
            label='0'
            #设置是否为根节点的标识
            root=[]
            #先添加根节点
            #可能存在查不到的情况，需要设置为0
            if target_number in user['phone_no_m'].values:
                node_tensor,label=self.extract_user_columns_to_tensor(user[user['phone_no_m'] == target_number],node_required_columns)
            else:
                node_tensor=torch.zeros(len(node_required_columns))
            nodes.append(node_tensor.type(torch.float32).reshape(-1,node_tensor.shape[-1]))
            root.append(1)
            for index, row in group_a.iterrows():
                #对上述列表逐个添加特征
                if row['opposite_no_m'] not in dic.keys():
                    dic[row['opposite_no_m']]=len(dic.keys())
                    #可能存在查不到的情况，需要设置为0
                    if row['opposite_no_m'] in user_op['phone_no_m'].values:
                        node_tensor,_=self.extract_user_columns_to_tensor(user_op[user_op['phone_no_m'] == row['opposite_no_m']],node_required_columns)
                    else:
                        node_tensor=torch.zeros(len(node_required_columns)).type(torch.float32)
                    
                    nodes.append(node_tensor.reshape(-1,node_tensor.shape[-1]).type(torch.float32))
                    root.append(0)
                edges.append([dic[target_number],dic[row['opposite_no_m']]])
                voc_tensor = self.extract_voc_columns_to_tensor(row,voc_required_columns).type(torch.float32)
                voc_tensors.append(voc_tensor[0,3:].reshape(-1,voc_tensor[0,3:].shape[-1]))
                poss.append(voc_tensor[0,:3].reshape(-1,3))
            if label==0:
                idx_non_fraud+=1
            else:
                idx_fraud+=1
            #统计获取边特征，并且去掉重复边
            edges,edge_tensors=self.trans(edges,voc_tensors)
            nodes=torch.cat(nodes,dim=0)
            voc_tensors=torch.cat(voc_tensors,dim=0)
            poss=torch.cat(poss,dim=0)
            label=torch.tensor(label)
            root=torch.tensor(root)
            #从数据中创建一个DATA对象
            data=Data(x=nodes,edge_index=edges,edge_attr=edge_tensors,voc_attr=voc_tensors,y=label,pos=poss,root=root)
            if idx_fraud==1:
                print(data)
            data_dic[str(target_number)+'.pt']=data
        return idx_non_fraud,idx_fraud,data_dic
    def get_users(self):
        users = pd.read_csv(self.raw_file_users, dtype='str')
        return users
    def cylindrical_to_cartesian(self, r, theta, h):
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = h
        return x, y, z
    def cacu_time(self,df):
        z = (datetime.datetime(int(df['year']), int(df['month']), int(df['day']))-self.start_date).days
        r = 1
        alltime=24*60*60
        now=int(df['hour'])*3600+int(df['minute'])*60+int(df['second'])
        theta = now/alltime*2*math.pi
        x, y, z = self.cylindrical_to_cartesian(r, theta, z)
        return x,y,z
    # 获取通话时间特征
    def get_voc_feat(self,df):
        print('正在处理时间列。。。')
        df["start_datetime"] = pd.to_datetime(df['start_datetime'])
        df["year"] = df["start_datetime"].dt.year
        df["month"] = df["start_datetime"].dt.month
        df["day"] = df["start_datetime"].dt.day
        df['hour'] = df["start_datetime"].dt.hour
        df['minute'] = df["start_datetime"].dt.minute
        df['second'] = df["start_datetime"].dt.second
        print('--------------已添加时间列！-----------')
        return df
    # 读取原始数据，对voc数据进行编码
    def get_voc_embed(self,original_df):
        #计算地址变化情况,注意需要确保话单是按照时间顺序的
        # 初始化新的列
        original_df['imei变化'] = '-1'
        original_df['城市变化'] = '-1'
        original_df['国家变化'] = '-1'
        original_df['x'] = '0'
        original_df['y'] = '0'
        original_df['z'] = '0'
        # 遍历数据并更新列
        print('正在遍历voc数据，并进行处理。。。')
        for i, row in tqdm(original_df.iterrows(), total=len(original_df)):
            # 对每行额外更新通话时长和坐标信息
            if original_df.at[i, 'calltype_id'] == '2':
                original_df.at[i, 'call_dur'] = -int(original_df.at[i, 'call_dur'])
            else:
                original_df.at[i, 'call_dur'] = int(original_df.at[i, 'call_dur'])
            original_df.at[i, 'x'],original_df.at[i, 'y'],original_df.at[i, 'z']=self.cacu_time(original_df.iloc[i])
            #必须要求是同一个电话号码才开始比较
            if i>0 and original_df.at[i, 'phone_no_m'] == original_df.at[i - 1, 'phone_no_m']:
                # 判断属性是否变化
                if pd.isna(original_df.at[i, 'imei_m']) or pd.isna(original_df.at[i - 1, 'imei_m']):
                    pass
                elif original_df.at[i, 'imei_m'] != original_df.at[i - 1, 'imei_m']:
                    original_df.at[i, 'imei变化'] = '1'
                if pd.isna(original_df.at[i, 'city_name']) or pd.isna(original_df.at[i - 1, 'city_name']):
                    pass
                elif original_df.at[i, 'city_name'] != original_df.at[i - 1, 'city_name']:
                    original_df.at[i, '城市变化'] = '1'
                if pd.isna(original_df.at[i, 'county_name']) or pd.isna(original_df.at[i - 1, 'county_name']):
                    pass
                elif original_df.at[i, 'county_name'] != original_df.at[i - 1, 'county_name']:
                    original_df.at[i, '国家变化'] = '1'

        #对给定的属性列获取独热编码并存储在'dummies'列
        encoded_df = pd.get_dummies(original_df, columns=['calltype_id','imei变化','城市变化','国家变化'], prefix=['calltype_id_dummies','imei变化_dummies','城市变化_dummies','国家变化_dummies'])
        original_df=original_df[['calltype_id','imei变化','城市变化','国家变化']].join(encoded_df)
        lower_bound = -1  # 下限
        upper_bound = 1   # 上限
        z_list=['call_dur']
        for name in z_list:
            original_df[name].fillna(0, inplace=True)
            original_df[name]=zscore(original_df[name].astype(float), nan_policy='omit')
            original_df[name] = np.clip(original_df[name], lower_bound, upper_bound)
        #把z轴控制在0,height的区间
        height=2
        original_df['z'] = (original_df['z'] - original_df['z'].min()) / (original_df['z'].max() - original_df['z'].min()) * height
        return original_df
    def get_and_transfer(self):
        # 获取通话时间信息 这步骤就是把通话的时间拆分成了小时，分钟等等单独的信息保存在了通话信息表中
        df = self.get_voc_feat(self.voc)
        #对df直接获取编码
        df=self.get_voc_embed(df)
        df = df.sort_values(by=['phone_no_m', 'start_datetime'])
        df.reset_index(drop=True, inplace=True)
        return df
    def get_data_obj(self):
        #目的是获取带有label的voc信息，把内部数据集的voc数据格式转化为公开数据集格式(额外多一行label信息)，然后直接复用代码
        voc_ori=self.get_and_transfer()
        #分别获得用户信息，并进行编码，处理na数据
        user_ori=self.get_users()
        z_list = ['sms_count','sms_nunique','sms_rate','calltype_2','calltype_rate','hour_mode','hour_mode_count','hour_nunique','day_mode','day_mode_count','day_nunique','busi_count','flow_mean','flow_median','flow_min','flow_max','flow_var','flow_sum','month_ids','flow_month','arpu_mean','arpu_var','arpu_max','arpu_min','arpu_median','arpu_sum','arpu_skew','arpu_sem','arpu_quantile','opposite_count','opposite_unique','voccalltype1','imeis','voc_calltype1','calltype_id_unique_x','phone2opposite_mean','phone2opposite_median','phone2opposite_max','phone2opposite_min','phone2opposite_var','phone2opposite_skew','phone2opposite_sem','phone2opposite_std','phone2opposite_quantile','phone2oppo_sum_mean','phone2oppo_sum_median','phone2oppo_sum_max','phone2oppo_sum_min','phone2oppo_sum_var','phone2oppo_sum_skew','phone2oppo_sum_sem','phone2oppo_sum_std','phone2oppo_sum_quantile','call_dur_mean','call_dur_median','call_dur_max','call_dur_min','call_dur_var','call_dur_skew','call_dur_sem','call_dur_std','call_dur_quantile','city_name_nunique','county_name_nunique','calltype_id_unique_y','voc_hour_mode','voc_hour_mode_count','voc_hour_nunique','voc_day_mode','voc_day_mode_count','voc_day_nunique']
        for name in z_list:
            user_ori[name].fillna(0, inplace=True)
        #对通话voc记录进行保存，保存成图需要的Data对象
        all={}
        print('正在处理graph信息获得obj对象')
        idx_non_fraud,idx_fraud,obj=self.get_obj(voc_ori,user_ori)
        all.update(obj)
        return idx_non_fraud,idx_fraud,all
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    def process(self):
        idx = 0
        exist_files=os.listdir(self.processed_dir+'_dataset6/')
        # 检查列表a是否全部包含在file_names中
        all_included = all(file_name in exist_files for file_name in [x+'.pt' for x in self.name])
        if not all_included:
            fraud_obj_num,non_fraud_obj_num,all_obj=self.get_data_obj()
            print('正在输出对象')
            print('共有'+str(fraud_obj_num)+'个诈骗对象，正在输出')
            print('共有'+str(non_fraud_obj_num)+'个正常对象，正在输出')
            for name in all_obj.keys():
                data = all_obj[name]

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir+'_dataset6/', name))
                idx += 1
        else:
            print('文件已经都处理好，无需进一步处理')
    def change_mode(self,mode):
        self.mode=mode
    def len(self):
        if self.mode=='train_name_01':
            return len(self.train_name_01)
        elif self.mode=='val_name_01':
            return len(self.val_name_01)
        elif self.mode=='test_name_01':
            return len(self.test_name_01)
        elif self.mode=='train_name_02':
            return len(self.train_name_02)
        elif self.mode=='val_name_02':
            return len(self.val_name_02)
        elif self.mode=='test_name_02':
            return len(self.test_name_02)
        elif self.mode=='train_name_04':
            return len(self.train_name_04)
        elif self.mode=='val_name_04':
            return len(self.val_name_04)
        elif self.mode=='test_name_04':
            return len(self.test_name_04)
        elif self.mode=='train_name_06':
            return len(self.train_name_06)
        elif self.mode=='val_name_06':
            return len(self.val_name_06)
        elif self.mode=='test_name_06':
            return len(self.test_name_06)
    def get_tranformer(self):
        name_list=[]
        if self.mode=='train_name_01'or self.mode=='val_name_01'or self.mode=='test_name_01':
            name_list=self.train_name_01
        elif self.mode=='train_name_02'or self.mode=='val_name_02'or self.mode=='test_name_02':
            name_list=self.train_name_02
        elif self.mode=='train_name_04'or self.mode=='val_name_04'or self.mode=='test_name_04':
            name_list=self.train_name_04
        elif self.mode=='train_name_06'or self.mode=='val_name_06'or self.mode=='test_name_06':
            name_list=self.train_name_06
        all_x=[]
        all_y=[]
        for n in name_list:
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', n.rstrip('.txt')+'.pt'))
            root_index= torch.nonzero(data.root).squeeze()
            feature=data.x[root_index].reshape(-1,data.x.shape[-1]).type(torch.float32)
            label=data.y
            all_x.append(feature)
            all_y.append(label)
        x=torch.cat(all_x,dim=0).cpu().numpy()
        y=torch.cat(all_y,dim=0).cpu().numpy()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit_transform(x)
    def get(self, idx):
        #需要进行一些数据的维度翻转（适应类定义）和存储名字更改（dataloader的batch会根据x识别，所以不能直接存储）
        if self.mode=='train_name_01':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.train_name_01[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='val_name_01':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.val_name_01[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='test_name_01':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.test_name_01[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='train_name_02':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.train_name_02[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='val_name_02':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.val_name_02[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='test_name_02':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.test_name_02[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='train_name_04':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.train_name_04[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='val_name_04':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.val_name_04[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='test_name_04':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.test_name_04[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='train_name_06':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.train_name_06[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='val_name_06':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.val_name_06[idx].rstrip('.txt')+'.pt'))
        elif self.mode=='test_name_06':
            data = torch.load(osp.join(self.processed_dir+'_dataset6/', self.test_name_06[idx].rstrip('.txt')+'.pt'))
        root_index= torch.nonzero(data.root).squeeze()
        feature=data.x[root_index].reshape(-1,data.x.shape[-1]).type(torch.float32)
        data.x_temp=data.x
        new = torch.tensor(self.scaler.transform(feature.cpu().numpy()))
        data.x_temp[root_index] = new
        data.x=None
        data.edge_index=data.edge_index.permute(1, 0)
        return data

#通话记录信息由统计获得，但是同时附带周级别特征和天级别特征
class Graph_Dataset8(Dataset):
    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.mode=mode

    @property
    def raw_file_names(self):
        self.raw_file_users=self.raw_dir
        self.raw_file_users_white=self.raw_dir+'/白名单用户.csv'
        self.raw_file_users_fraud=self.raw_dir+'/纯涉诈卡.csv'
        self.raw_file_users_mutation=self.raw_dir+'/突变涉诈卡.csv'
        self.raw_file_white_opusers=self.raw_dir+'/白名单对端号码.csv'
        self.raw_file_fraud_opusers=self.raw_dir+'/纯涉诈卡对端号码.csv'
        self.raw_file_change_opusers=self.raw_dir+'/突变涉诈卡对端号码.csv'
        self.raw_file_white_voc=self.raw_dir+'/白名单话单.csv'
        self.raw_file_fraud_voc=self.raw_dir+'/纯涉诈卡话单.csv'
        self.raw_file_change_voc=self.raw_dir+'/突变涉诈卡话单.csv'
        return [self.raw_file_users,self.raw_file_white_opusers,self.raw_file_fraud_opusers,self.raw_file_change_opusers,self.raw_file_white_voc,self.raw_file_fraud_voc,self.raw_file_change_voc]

    @property
    def processed_file_names(self):
        self.start_date = datetime.datetime(2023,8,1)   # 第一个日期
        self.ratio=[0.6,0.2,0.2]
        self.train_name=[]
        self.val_name=[]
        self.val_name=[]
        self.voc_fraud = pd.read_csv(self.raw_file_fraud_voc, dtype='str')
        self.voc_change = pd.read_csv(self.raw_file_change_voc, dtype='str')
        self.voc_fraud['label'] = 1
        self.voc_change['label'] = 1
        print('纯诈骗话单读取矩阵形状：')
        print(self.voc_fraud.shape)
        print('突变诈骗话单读取矩阵形状：')
        print(self.voc_change.shape)
        self.voc_non_fraud = pd.read_csv(self.raw_file_white_voc, dtype='str')
        self.voc_non_fraud['label'] = 0
        print('非诈骗话单读取矩阵形状：')
        print(self.voc_non_fraud.shape)

        idx=0
        voc_fraud_name=[]
        for x in self.voc_fraud['号码'].unique():
            voc_fraud_name.append('fraud_'+str(idx)+'.pt')
            idx+=1
        idx=0
        voc_change_name=[]
        for x in self.voc_change['号码'].unique():
            voc_change_name.append('change_'+str(idx)+'.pt')
            idx+=1
        idx=0
        voc_non_fraud_name=[]
        for x in self.voc_non_fraud['号码'].unique():
            voc_non_fraud_name.append('non_fraud_'+str(idx)+'.pt')
            idx+=1
        all=voc_fraud_name+voc_non_fraud_name+voc_change_name
        #划分数据集
        train_fraud = int(self.ratio[0] * len(voc_fraud_name))
        val_fraud = int(self.ratio[1] * len(voc_fraud_name))
        train_change = int(self.ratio[0] * len(voc_change_name))
        val_change = int(self.ratio[1] * len(voc_change_name))
        train_non_fraud = int(self.ratio[0] * len(voc_non_fraud_name))
        val_non_fraud = int(self.ratio[1] * len(voc_non_fraud_name))
        # 设置随机种子
        random_seed = 0
        random.seed(random_seed)
        random.shuffle(voc_fraud_name)
        random.shuffle(voc_change_name)
        random.shuffle(voc_non_fraud_name)
        self.train_name=voc_fraud_name[:train_fraud]+voc_non_fraud_name[:train_non_fraud]+voc_change_name[:train_change]
        self.val_name=voc_fraud_name[train_fraud:train_fraud+val_fraud]+voc_non_fraud_name[train_non_fraud:train_non_fraud+val_non_fraud]+voc_change_name[train_change:train_change+val_change]
        self.test_name=voc_fraud_name[train_fraud+val_fraud:]+voc_non_fraud_name[train_non_fraud+val_non_fraud:]+voc_change_name[train_change+val_change:]
        self.val_change_name=voc_change_name[train_change:train_change+val_change]
        self.test_change_name=voc_change_name[train_change+val_change:]
        self.val_fraud_name=voc_fraud_name[train_fraud:train_fraud+val_fraud]
        self.test_fraud_name=voc_fraud_name[train_fraud+val_fraud:]
        print('train,val,test:',len(self.train_name),len(self.val_name),len(self.test_name))
        self.all_processed_files=all
        return [self.processed_dir+'_dataset8/'+x for x in all]
        #return [self.processed_dir+'/'+'aa.pt']
    # 自定义函数：根据字符串长度提取不同位置的字符
    
    def extract_user_columns_to_tensor(self,row,required_columns):
        # 检查DataFrame中是否包含所需的列名
        missing_columns = [col for col in required_columns if col not in row.columns]

        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # 抽取所需的列值
        extracted_values = row[required_columns].values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1)

        return tensor
    def extract_voc_columns_to_tensor(self,row):
        # 抽取所需的列值
        row.drop(columns=['号码'], inplace=True)
        extracted_values = row.values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1)

        return tensor
    def trans(self, index):
        #先进行字典初始化
        dic={}
        index_new=[]
        for edge in index:
            sub_tuple = tuple(sorted(edge))
            if sub_tuple not in dic.keys():
                dic[sub_tuple]=[]
            else:
                pass
            
        for edge in dic.keys():
            index_new.append(edge)

        # 将结果列表转换回张量
        index_new=torch.tensor(index_new)
        return index_new
    def get_obj(self,voc,user,user_op,name,prefix,voc_new):
        #先进行数据分组
        grouped = voc.groupby('号码')
        data_dic={}
        idx_non_fraud=0
        idx_fraud=0
        idx_change=0
        node_dummies_columns = user.filter(like='dummies', axis=1)
        node_required_columns = ['年龄','本月消费','本月使用流量/M','本月通话时长','3个月主叫占比','3个月主叫离散','3个月主叫外省号码占比']
        node_required_columns.extend(node_dummies_columns)
        for target_number in tqdm(name, desc="Processing"):
            group_a = grouped.get_group(target_number)  # 提取名为target_number的分组数据
            #存储号码序号的映射字典
            dic={target_number:0}
            nodes=[]
            edges=[]
            label='0'
            #设置是否为根节点的标识
            root=[]
            #先添加根节点
            #可能存在查不到的情况，需要设置为0
            if target_number in user['号码'].values:
                node_tensor=self.extract_user_columns_to_tensor(user[user['号码'] == target_number],node_required_columns)
            else:
                node_tensor=torch.zeros(len(node_required_columns))
            voc_tensor=self.extract_voc_columns_to_tensor(voc_new[voc_new['号码'] == target_number])
            node_tensor=torch.cat([voc_tensor,node_tensor.reshape(-1,node_tensor.shape[-1])],dim=-1)
            nodes.append(node_tensor.type(torch.float32).reshape(-1,node_tensor.shape[-1]))
            root.append(1)
            for index, row in group_a.iterrows():
                #对上述列表逐个添加特征
                if row['对端号码'] not in dic.keys():
                    dic[row['对端号码']]=len(dic.keys())
                    #可能存在查不到的情况，需要设置为0
                    if row['对端号码'] in user_op['号码'].values:
                        node_tensor=self.extract_user_columns_to_tensor(user_op[user_op['号码'] == row['对端号码']],node_required_columns)
                    else:
                        node_tensor=torch.zeros(len(node_required_columns)).type(torch.float32)
                    node_tensor=torch.cat([torch.zeros_like(voc_tensor),node_tensor.reshape(-1,node_tensor.shape[-1])],dim=-1)
                    nodes.append(node_tensor.reshape(-1,node_tensor.shape[-1]).type(torch.float32))
                    root.append(0)
                edges.append([dic[target_number],dic[row['对端号码']]])
                label=row['label']
            #统计获取边特征，并且去掉重复边
            edges=self.trans(edges)
            nodes=torch.cat(nodes,dim=0)
            label=torch.tensor(label)
            root=torch.tensor(root)
            #从数据中创建一个DATA对象
            data=Data(x=nodes,edge_index=edges,y=label,root=root)
            if prefix=='non_fraud_':
                data_dic['non_fraud_'+str(idx_non_fraud)+'.pt']=data
                idx_non_fraud+=1
            elif prefix=='fraud_':
                data_dic['fraud_'+str(idx_fraud)+'.pt']=data
                idx_fraud+=1
            elif prefix=='change_':
                data_dic['change_'+str(idx_change)+'.pt']=data
                idx_change+=1
        return data_dic
    def get_users(self,data_path):
        user_fraud = pd.read_csv(self.raw_file_users_fraud, dtype='str')
        user_change = pd.read_csv(self.raw_file_users_mutation, dtype='str')
        user_non_fraud = pd.read_csv(self.raw_file_users_white, dtype='str')
        user_fraud_opusers = pd.read_csv(self.raw_file_fraud_opusers, dtype='str')
        user_change_opusers = pd.read_csv(self.raw_file_change_opusers, dtype='str')
        user_non_fraud_opusers = pd.read_csv(self.raw_file_white_opusers, dtype='str')
        all=[user_fraud , user_change, user_non_fraud,user_fraud_opusers,user_change_opusers,user_non_fraud_opusers]
        length=[len(x) for x in all]
        for i in range(1,len(length)):
            length[i]=length[i]+length[i-1]
        # 在行方向上拼接数据
        df = pd.concat(all, ignore_index=True)
        df=self.get_user_embed(df)
        return df[:length[0]],df[length[0]:length[1]],df[length[1]:length[2]],df[length[2]:length[3]],df[length[3]:length[4]],df[length[4]:length[5]],df[:length[2]]
    def get_user_embed(self,original_df):
        #对给定的属性列获取独热编码并存储在'hot'
        encoded_df = pd.get_dummies(original_df, columns=['开户接入方式','宽带标识'], prefix="dummies")
        original_df=original_df[['开户接入方式','宽带标识']].join(encoded_df)
        z_list=['年龄','本月消费','本月使用流量/M','本月通话时长','3个月主叫占比','3个月主叫离散','3个月主叫外省号码占比']
        for name in z_list:
            original_df[name].fillna(0, inplace=True)
        return original_df
    def extract_hours(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[:2]
    # 自定义函数：根据字符串长度提取不同位置的字符
    def extract_minutes(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[2:4]
    # 自定义函数：根据字符串长度提取不同位置的字符
    def extract_seconds(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[4:]
    def cylindrical_to_cartesian(self, r, theta, h):
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = h
        return x, y, z
    def cacu_time(self,df):
        z = (datetime.datetime(int(df['year']), int(df['month']), int(df['day']))-self.start_date).days
        r = 1
        alltime=24*60*60
        now=int(df['hour'])*3600+int(df['minute'])*60+int(df['second'])
        theta = now/alltime*2*math.pi
        x, y, z = self.cylindrical_to_cartesian(r, theta, z)
        return x,y,z
    def get_voc_day(self,df):
        print('开始更细粒度处理通话数据')
        df["start_datetime"] = pd.to_datetime(df['日期'])
        df["hour"] = df['时间'].apply(self.extract_hours).astype('int64')
        df["day"] = df['start_datetime'].dt.day
        df["通话时长"] = df['通话时长'].astype('int64')
        phone_no_m = df[["号码"]].copy()
        phone_no_m = phone_no_m.drop_duplicates(subset=['号码'], keep='last')
        # 对话人数和对话次数
        all=[]
        print('正在计算每天通话次数和通话人数的相关统计信息')
        tmp = df.groupby(["号码","day"])["对端号码"].agg(opposite_count="count", opposite_unique="nunique")
        phone2opposite1 = tmp.groupby("号码")["opposite_count"].agg(day_opposite_count_mean="mean"
                                                                , day_opposite_count_median="median"
                                                                , day_opposite_count_max="max"
                                                                , day_opposite_count_min="min"
                                                                , day_opposite_count_var="var"
                                                                , day_opposite_count_skew="skew"
                                                                , day_opposite_count_sem="sem"
                                                                , day_opposite_count_std="std"
                                                                , day_opposite_count_quantile="quantile"
                                                                )
        all.append(phone2opposite1)
        phone2opposite2 = tmp.groupby("号码")["opposite_unique"].agg(day_opposite_unique_mean="mean"
                                                            , day_opposite_unique_median="median"
                                                            , day_opposite_unique_max="max"
                                                            , day_opposite_unique_min="min"
                                                            , day_opposite_unique_var="var"
                                                            , day_opposite_unique_skew="skew"
                                                            , day_opposite_unique_sem="sem"
                                                            , day_opposite_unique_std="std"
                                                            , day_opposite_unique_quantile="quantile"
                                                            )
        all.append(phone2opposite2)
        tmp = df.groupby(["号码","day"])["通话时长"].agg(day_sum="sum")
        day_call = tmp.groupby("号码")["day_sum"].agg(day_call_dur_mean="mean"
                                                    , day_call_dur_median="median"
                                                    , day_call_dur_max="max"
                                                    , day_call_dur_min="min"
                                                    , day_call_dur_var="var"
                                                    , day_call_dur_skew="skew"
                                                    , day_call_dur_sem="sem"
                                                    , day_call_dur_std="std"
                                                    , day_call_dur_quantile="quantile"
                                                    )
        all.append(day_call)
        return all
    def get_voc_week(self,df):
        print('开始更细粒度处理通话数据')
        df["start_datetime"] = pd.to_datetime(df['日期'])
        df["hour"] = df['时间'].apply(self.extract_hours).astype('int64')
        df["year"] = df['start_datetime'].dt.year
        df["month"] = df['start_datetime'].dt.month
        df["day"] = df['start_datetime'].dt.day
        df["week"] = ((datetime.datetime(int(df['year']), int(df['month']), int(df['day']))-self.start_date).days)//7

        df["通话时长"] = df['通话时长'].astype('int64')
        phone_no_m = df[["号码"]].copy()
        phone_no_m = phone_no_m.drop_duplicates(subset=['号码'], keep='last')
        # 对话人数和对话次数
        all=[]
        print('正在计算每天通话次数和通话人数的相关统计信息')
        tmp = df.groupby(["号码",'week'])["对端号码"].agg(opposite_count="count", opposite_unique="nunique")
        phone2opposite1 = tmp.groupby("号码")["opposite_count"].agg(week_opposite_count_mean="mean"
                                                                , week_opposite_count_median="median"
                                                                , week_opposite_count_max="max"
                                                                , week_opposite_count_min="min"
                                                                , week_opposite_count_var="var"
                                                                , week_opposite_count_skew="skew"
                                                                , week_opposite_count_sem="sem"
                                                                , week_opposite_count_std="std"
                                                                , week_opposite_count_quantile="quantile"
                                                                )
        all.append(phone2opposite1)
        phone2opposite2 = tmp.groupby("号码")["opposite_unique"].agg(week_opposite_unique_mean="mean"
                                                            , week_opposite_unique_median="median"
                                                            , week_opposite_unique_max="max"
                                                            , week_opposite_unique_min="min"
                                                            , week_opposite_unique_var="var"
                                                            , week_opposite_unique_skew="skew"
                                                            , week_opposite_unique_sem="sem"
                                                            , week_opposite_unique_std="std"
                                                            , week_opposite_unique_quantile="quantile"
                                                            )
        all.append(phone2opposite2)
        tmp = df.groupby(["号码","week"])["通话时长"].agg(week_sum="sum")
        week_call = tmp.groupby("号码")["week_sum"].agg(week_call_dur_mean="mean"
                                                    , week_call_dur_median="median"
                                                    , week_call_dur_max="max"
                                                    , week_call_dur_min="min"
                                                    , week_call_dur_var="var"
                                                    , week_call_dur_skew="skew"
                                                    , week_call_dur_sem="sem"
                                                    , week_call_dur_std="std"
                                                    , week_call_dur_quantile="quantile"
                                                    )
        all.append(week_call)
        return all
    def get_voc_feat(self, df):

        #获得周级别和天级别的通话统计信息
        day=self.get_voc_day(df)
        week=self.get_voc_day(df)

        print('开始处理通话数据')
        df["start_datetime"] = pd.to_datetime(df['日期'])
        df["hour"] = df['时间'].apply(self.extract_hours).astype('int64')
        df["day"] = df['start_datetime'].dt.day
        df["通话时长"] = df['通话时长'].astype('int64')
        phone_no_m = df[["号码"]].copy()
        phone_no_m = phone_no_m.drop_duplicates(subset=['号码'], keep='last')
        # 对话人数和对话次数
        print('正在计算通话次数和通话人数')
        tmp = df.groupby("号码")["对端号码"].agg(opposite_count="count", opposite_unique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")
        """主叫通话
        """
        print('正在处理通话类型为主叫的电话信息：')
        print('     正在计算imeis个数')
        df_call = df[df["呼叫类型"] == '1'].copy()
        tmp = df_call.groupby("号码")["imei"].agg(voccalltype1="count", imeis="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")
        print('     正在计算主叫占比')
        phone_no_m["voc_calltype1"] = phone_no_m["voccalltype1"] / phone_no_m["opposite_count"]
        print('     正在计算通话类型个数')
        tmp = df.groupby("号码")["呼叫类型"].agg(calltype_id_unique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")

        """和固定通话者的对话统计
        """
        print('正在统计通话交互行为信息:')

        tmp = df.groupby(["号码", "对端号码"])["通话时长"].agg(count="count", sum="sum")
        print('     正在统计和固定通话者的通话次数信息')
        phone2opposite = tmp.groupby("号码")["count"].agg(phone2opposite_mean="mean"
                                                                , phone2opposite_median="median"
                                                                , phone2opposite_max="max"
                                                                , phone2opposite_min="min"
                                                                , phone2opposite_var="var"
                                                                , phone2opposite_skew="skew"
                                                                , phone2opposite_sem="sem"
                                                                , phone2opposite_std="std"
                                                                , phone2opposite_quantile="quantile"
                                                                )

        phone_no_m = phone_no_m.merge(phone2opposite, on="号码", how="left")
        print('     正在统计和固定通话者的通话总时长信息')
        phone2opposite = tmp.groupby("号码")["sum"].agg(phone2oppo_sum_mean="mean"
                                                            , phone2oppo_sum_median="median"
                                                            , phone2oppo_sum_max="max"
                                                            , phone2oppo_sum_min="min"
                                                            , phone2oppo_sum_var="var"
                                                            , phone2oppo_sum_skew="skew"
                                                            , phone2oppo_sum_sem="sem"
                                                            , phone2oppo_sum_std="std"
                                                            , phone2oppo_sum_quantile="quantile"
                                                            )

        phone_no_m = phone_no_m.merge(phone2opposite, on="号码", how="left")

        """通话时间长短统计
        """
        print('     正在统计和固定通话者的每次通话时长信息')
        tmp = df.groupby("号码")["通话时长"].agg(call_dur_mean="mean"
                                                    , call_dur_median="median"
                                                    , call_dur_max="max"
                                                    , call_dur_min="min"
                                                    , call_dur_var="var"
                                                    , call_dur_skew="skew"
                                                    , call_dur_sem="sem"
                                                    , call_dur_std="std"
                                                    , call_dur_quantile="quantile"
                                                    )
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")

        tmp = df.groupby("号码")["对端号码归属"].agg(city_name_nunique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")
        tmp = df.groupby("号码")["呼叫类型"].agg(calltype_id_unique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")

        """通话时间点偏好
        """
        print('正在处理通话时间偏好信息：')
        print('     正在计算每日最常通话时间点，及在该时间点通话次数，通话时间分布')
        tmp = df.groupby("号码")["hour"].agg(voc_hour_mode=lambda x: stats.mode(x)[0][0],
                                                voc_hour_mode_count=lambda x: stats.mode(x)[1][0],
                                                voc_hour_nunique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")
        print('     正在计算每月最常通话日期，及在该日期通话次数，通话时间分布')
        tmp = df.groupby("号码")["day"].agg(voc_day_mode=lambda x: stats.mode(x)[0][0],
                                                voc_day_mode_count=lambda x: stats.mode(x)[1][0],
                                                voc_day_nunique="nunique")
        phone_no_m = phone_no_m.merge(tmp, on="号码", how="left")

        #将周级别的统计信息和天统计信息并列加入
        print(phone_no_m.shape)
        for t in week:
            phone_no_m = phone_no_m.merge(t, on="号码", how="left")
        print('week',phone_no_m.shape)
        for t in day:
            phone_no_m = phone_no_m.merge(t, on="号码", how="left")
        print('day',phone_no_m.shape)

        phone_no_m.fillna(0, inplace=True)
        z_list = phone_no_m.select_dtypes(include=['number'])
        for name in z_list:
            phone_no_m[name].fillna(0, inplace=True)
        return phone_no_m
    def get_and_transfer(self,data_path):
        return self.get_voc_feat(self.voc_fraud),self.get_voc_feat(self.voc_change),self.get_voc_feat(self.voc_non_fraud)
    def get_data_obj(self,data_path):
        #目的是获取带有label的voc信息
        voc_fraud_new,voc_change_new,voc_non_fraud_new=self.get_and_transfer(data_path)
        
        #分别获得用户信息，并进行编码
        user_fraud,user_change,user_non_fraud,user_fraud_op,user_change_op,user_non_fraud_op,user_ori=self.get_users(data_path)
        
        fraud_name=voc_fraud_new['号码'].unique()
        change_name=voc_change_new['号码'].unique()
        non_fraud_name=voc_non_fraud_new['号码'].unique()

        #对通话voc记录进行保存，保存成图需要的Data对象
        all={}
        print('正在处理graph信息获得obj对象')
        fraud_obj=self.get_obj(self.voc_fraud,user_fraud,user_fraud_op,fraud_name,'fraud_',voc_fraud_new)
        change_obj=self.get_obj(self.voc_change,user_change,user_change_op,change_name,'change_',voc_change_new)
        #dic obj,需要用名字作为key以便后续存储
        non_fraud_obj=self.get_obj(self.voc_non_fraud,user_non_fraud,user_non_fraud_op,non_fraud_name,'non_fraud_',voc_non_fraud_new)
        all.update(fraud_obj)
        all.update(change_obj)
        all.update(non_fraud_obj)
        return fraud_obj,change_obj,non_fraud_obj,all
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    def process(self):
        idx = 0
        exist_files=os.listdir(self.processed_dir+'_dataset8/')
        # 检查列表a是否全部包含在file_names中
        all_included = all(file_name in exist_files for file_name in self.all_processed_files)
        if not all_included:
            fraud_obj,change_obj,non_fraud_obj,all_obj=self.get_data_obj(self.raw_file_users)
            print('共有'+str(len(fraud_obj))+'个诈骗对象，正在输出')
            print('共有'+str(len(change_obj))+'个突变诈骗对象，正在输出')
            print('共有'+str(len(non_fraud_obj))+'个正常对象，正在输出')
            for name in all_obj.keys():
                data = all_obj[name]

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir+'_dataset8/', name))
                idx += 1
        else:
            print('文件已经都处理好，无需进一步处理')
    def change_mode(self,mode):
        self.mode=mode
    def len(self):
        if self.mode=='train':
            return len(self.train_name)
        elif self.mode=='val':
            return len(self.val_name)
        elif self.mode=='test':
            return len(self.test_name)
        elif self.mode=='change':
            return len(self.test_change_name)
        elif self.mode=='fraud':
            return len(self.test_fraud_name)
    def get_tranformer(self):
        name_list=self.train_name
        all_x=[]
        all_y=[]
        for n in name_list:
            data = torch.load(osp.join(self.processed_dir+'_dataset8/', n))
            root_index= torch.nonzero(data.root).squeeze()
            feature=data.x[root_index].reshape(-1,data.x.shape[-1]).type(torch.float32)
            label=data.y
            all_x.append(feature)
            all_y.append(label)
        x=torch.cat(all_x,dim=0).cpu().numpy()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit_transform(x)
    def get(self, idx):
        #需要进行一些数据的维度翻转（适应类定义）和存储名字更改（dataloader的batch会根据x识别，所以不能直接存储）
        if self.mode=='train':
            data = torch.load(osp.join(self.processed_dir+'_dataset8/', self.train_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='val':
            data = torch.load(osp.join(self.processed_dir+'_dataset8/', self.val_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='test':
            data = torch.load(osp.join(self.processed_dir+'_dataset8/', self.test_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='change':
            data = torch.load(osp.join(self.processed_dir+'_dataset8/', self.test_change_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)
        elif self.mode=='fraud':
            data = torch.load(osp.join(self.processed_dir+'_dataset8/', self.test_fraud_name[idx]))
            data.x_temp=torch.tensor(self.scaler.transform(data.x.cpu().numpy()))
            data.x=None
            data.num_nodes=data.x_temp.shape[0]
            data.edge_index=data.edge_index.permute(1, 0)

        return data

#加载数据集，用来实现公开方法cdr2img
class Graph_Dataset9(Dataset):
    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.mode=mode

    @property
    def raw_file_names(self):
        self.raw_file_users=self.raw_dir
        self.raw_file_users_white=self.raw_dir+'/白名单用户.csv'
        self.raw_file_users_fraud=self.raw_dir+'/纯涉诈卡.csv'
        self.raw_file_users_mutation=self.raw_dir+'/突变涉诈卡.csv'
        self.raw_file_white_opusers=self.raw_dir+'/白名单对端号码.csv'
        self.raw_file_fraud_opusers=self.raw_dir+'/纯涉诈卡对端号码.csv'
        self.raw_file_change_opusers=self.raw_dir+'/突变涉诈卡对端号码.csv'
        self.raw_file_white_voc=self.raw_dir+'/白名单话单.csv'
        self.raw_file_fraud_voc=self.raw_dir+'/纯涉诈卡话单.csv'
        self.raw_file_change_voc=self.raw_dir+'/突变涉诈卡话单.csv'
        return [self.raw_file_users,self.raw_file_white_opusers,self.raw_file_fraud_opusers,self.raw_file_change_opusers,self.raw_file_white_voc,self.raw_file_fraud_voc,self.raw_file_change_voc]

    @property
    def processed_file_names(self):
        self.start_date = datetime.datetime(2023,8,1)   # 第一个日期
        self.ratio=[0.6,0.2,0.2]
        self.train_name=[]
        self.val_name=[]
        self.val_name=[]
        self.voc_fraud = pd.read_csv(self.raw_file_fraud_voc, dtype='str')
        self.voc_change = pd.read_csv(self.raw_file_change_voc, dtype='str')
        self.voc_fraud['label'] = 1
        self.voc_change['label'] = 1
        print('纯诈骗话单读取矩阵形状：')
        print(self.voc_fraud.shape)
        print('突变诈骗话单读取矩阵形状：')
        print(self.voc_change.shape)
        self.voc_non_fraud = pd.read_csv(self.raw_file_white_voc, dtype='str')
        self.voc_non_fraud['label'] = 0
        print('非诈骗话单读取矩阵形状：')
        print(self.voc_non_fraud.shape)

        idx=0
        voc_fraud_name=[]
        for x in self.voc_fraud['号码'].unique():
            voc_fraud_name.append('fraud_'+str(idx)+'.pt')
            idx+=1
        idx=0
        voc_change_name=[]
        for x in self.voc_change['号码'].unique():
            voc_change_name.append('change_'+str(idx)+'.pt')
            idx+=1
        idx=0
        voc_non_fraud_name=[]
        for x in self.voc_non_fraud['号码'].unique():
            voc_non_fraud_name.append('non_fraud_'+str(idx)+'.pt')
            idx+=1
        all=voc_fraud_name+voc_non_fraud_name+voc_change_name
        #划分数据集
        train_fraud = int(self.ratio[0] * len(voc_fraud_name))
        val_fraud = int(self.ratio[1] * len(voc_fraud_name))
        train_change = int(self.ratio[0] * len(voc_change_name))
        val_change = int(self.ratio[1] * len(voc_change_name))
        train_non_fraud = int(self.ratio[0] * len(voc_non_fraud_name))
        val_non_fraud = int(self.ratio[1] * len(voc_non_fraud_name))
        # 设置随机种子
        random_seed = 0
        random.seed(random_seed)
        random.shuffle(voc_fraud_name)
        random.shuffle(voc_change_name)
        random.shuffle(voc_non_fraud_name)
        self.train_name=voc_fraud_name[:train_fraud]+voc_non_fraud_name[:train_non_fraud]+voc_change_name[:train_change]
        self.val_name=voc_fraud_name[train_fraud:train_fraud+val_fraud]+voc_non_fraud_name[train_non_fraud:train_non_fraud+val_non_fraud]+voc_change_name[train_change:train_change+val_change]
        self.test_name=voc_fraud_name[train_fraud+val_fraud:]+voc_non_fraud_name[train_non_fraud+val_non_fraud:]+voc_change_name[train_change+val_change:]
        self.val_change_name=voc_change_name[train_change:train_change+val_change]
        self.test_change_name=voc_change_name[train_change+val_change:]
        self.val_fraud_name=voc_fraud_name[train_fraud:train_fraud+val_fraud]
        self.test_fraud_name=voc_fraud_name[train_fraud+val_fraud:]
        print('train,val,test:',len(self.train_name),len(self.val_name),len(self.test_name))
        self.all_processed_files=all

        return [self.processed_dir+'_dataset9/'+x for x in all]
    # 自定义函数：根据字符串长度提取不同位置的字符
    
    def extract_user_columns_to_tensor(self,row,required_columns):
        # 检查DataFrame中是否包含所需的列名
        missing_columns = [col for col in required_columns if col not in row.columns]

        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # 抽取所需的列值
        extracted_values = row[required_columns].values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1)

        return tensor
    def extract_voc_columns_to_tensor(self,row,required_columns):
        # 检查DataFrame中是否包含所需的列名
        missing_columns = [col for col in required_columns if col not in row.index.tolist()]

        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # 抽取所需的列值
        extracted_values = row[required_columns].values.astype(float)
        # 创建PyTorch Tensor
        tensor = torch.tensor(extracted_values).reshape(1,-1)

        return tensor
    def trans(self, index, attr):
        #先进行字典初始化
        dic={}
        index_new=[]
        attr_new=[]
        for edge,t in zip(index,attr):
            sub_tuple = tuple(sorted(edge))
            if sub_tuple not in dic.keys():
                dic[sub_tuple]=[]
                dic[sub_tuple].append(t)
            else:
                dic[sub_tuple].append(t)
            
        #进行边特征计算
        for edge in dic.keys():
            index_new.append(edge)
            # 计算张量的平均值
            mean_tensor = torch.stack(dic[edge]).mean(dim=0)
            # 计算总张量个数
            total_tensors = len(dic[edge])
            # 将总张量个数添加到平均张量中
            mean_tensor_with_count = torch.cat((mean_tensor, torch.tensor([[total_tensors]])), dim=1)
            attr_new.append(mean_tensor_with_count.type(torch.float32))

        # 将结果列表转换回张量
        index_new=torch.tensor(index_new)
        attr_new=torch.cat(attr_new,dim=0)
        return index_new,attr_new
        # 先尝试形成一个number的矩阵
    def one_phone_number(self,df,mode='train'):
        # nf为填数矩阵，根据nf中opposite_count字段的值对矩阵进行填数
        # 求opposite_count这列的数量 总之，这段代码的目的是在原始 DataFrame 中添加一个新的列，该列包含了根据 phone_no_m、opposite_no_m 和 calltype_id 分组计数后的结果。通过将聚合的结果 DataFrame 与原始 DataFrame 进行合并，将计数信息添加到了每个对应的行中。
        nf =df.groupby(["号码","对端号码",'呼叫类型'])['呼叫类型'].agg(opposite_count="count")
        df = df.merge(nf,on=["号码","对端号码",'呼叫类型'])
        if mode=='test':
            d1 = datetime.datetime(2023,8,1)   # 第一个日期
            d2 = datetime.datetime(2023,9,14)   # 第二个日期
        else:
            d1 = datetime.datetime(2023,8,1)   # 第一个日期
            d2 = datetime.datetime(2023,9,14)   # 第二个日期
        interval = d2 - d1                   # 两日期差距
        days = interval.days+1  #矩阵的列
        hours = 24
        #     初始化矩阵
        call_matrix = np.zeros((hours, days))
        for m in range(df.shape[0]):
            # 将主叫置为正值，被叫置为负值
            if df.iloc[m]['呼叫类型'] == 1:
                df.loc[m,'通话时长'] = int(df.iloc[m]['通话时长'])
            else:
                df.loc[m,'通话时长'] = -int(df.iloc[m]['通话时长'])
        # 计算每小时通话时间
        tmp = df.groupby(["year", 'day','month','hour'])["通话时长"].agg(call_dur_sum='sum')
        #print(tmp)
        df = df.merge(tmp,how='left',on=["year", 'day','month','hour'])

        for m in range(df.shape[0]):
            year = df.iloc[m]['year']
            day = df.iloc[m]['day']
            month = df.iloc[m]['month']
            hour = df.iloc[m]['hour']
            call_dur_sum = df.iloc[m]['call_dur_sum']
            d3 = datetime.datetime(year, month, day)
            # 计算当前计算的日期距离初始日期的位置
            interval = d3 - d1
            column_index = interval.days
            if call_dur_sum > 0:
                call_matrix[hour,column_index] = 1
            elif call_dur_sum < 0:
                call_matrix[hour,column_index] = -1
        call_matrix=np.array(call_matrix)
        call_matrix = torch.tensor(call_matrix)
        return call_matrix
    def get_obj(self,voc,user,user_op,name,prefix):
        #先进行数据分组
        grouped = voc.groupby('号码')
        data_dic={}
        idx_non_fraud=0
        idx_fraud=0
        idx_change=0
        node_dummies_columns = user.filter(like='dummies', axis=1)
        node_required_columns = ['年龄','本月消费','本月使用流量/M','本月通话时长','3个月主叫占比','3个月主叫离散','3个月主叫外省号码占比']
        node_required_columns.extend(node_dummies_columns)
        voc_dummies_columns = voc.filter(like='dummies', axis=1)
        voc_required_columns = ['x','y','z','通话时长']
        voc_required_columns.extend(voc_dummies_columns)
        for target_number in tqdm(name, desc="Processing"):
            group_a = grouped.get_group(target_number)  # 提取名为target_number的分组数据
            data_i = voc[voc['号码'] == target_number]
            call_matrix = self.one_phone_number(data_i)
            #存储号码序号的映射字典
            dic={target_number:0}
            nodes=[]
            edges=[]
            voc_tensors=[]
            poss=[]
            label='0'
            #设置是否为根节点的标识
            root=[]
            #先添加根节点
            #可能存在查不到的情况，需要设置为0
            if target_number in user['号码'].values:
                node_tensor=self.extract_user_columns_to_tensor(user[user['号码'] == target_number],node_required_columns)
            else:
                node_tensor=torch.zeros(len(node_required_columns))
            nodes.append(node_tensor.type(torch.float32).reshape(-1,node_tensor.shape[-1]))
            root.append(1)
            for index, row in group_a.iterrows():
                #对上述列表逐个添加特征
                if row['对端号码'] not in dic.keys():
                    dic[row['对端号码']]=len(dic.keys())
                    #可能存在查不到的情况，需要设置为0
                    if row['对端号码'] in user_op['号码'].values:
                    #     node_tensor=self.extract_user_columns_to_tensor(user_op[user_op['号码'] == row['对端号码']],node_required_columns)
                    # else:
                        node_tensor=torch.zeros(len(node_required_columns)).type(torch.float32)
                    
                    nodes.append(node_tensor.reshape(-1,node_tensor.shape[-1]).type(torch.float32))
                    root.append(0)
                label=row['label']

            nodes=torch.cat(nodes,dim=0)
            label=torch.tensor(label)
            root=torch.tensor(root)
            #从数据中创建一个DATA对象
            data=Data(x=nodes,edge_index=None,edge_attr=None,voc_attr=call_matrix,y=label,pos=None,root=root)
            if idx_fraud==0:
                print(data)
            if prefix=='non_fraud_':
                data_dic['non_fraud_'+str(idx_non_fraud)+'.pt']=data
                idx_non_fraud+=1
            elif prefix=='fraud_':
                data_dic['fraud_'+str(idx_fraud)+'.pt']=data
                idx_fraud+=1
            elif prefix=='change_':
                data_dic['change_'+str(idx_change)+'.pt']=data
                idx_change+=1
        return data_dic
    def get_users(self,data_path):
        user_fraud = pd.read_csv(self.raw_file_users_fraud, dtype='str')
        user_change = pd.read_csv(self.raw_file_users_mutation, dtype='str')
        user_non_fraud = pd.read_csv(self.raw_file_users_white, dtype='str')
        user_fraud_opusers = pd.read_csv(self.raw_file_fraud_opusers, dtype='str')
        user_change_opusers = pd.read_csv(self.raw_file_change_opusers, dtype='str')
        user_non_fraud_opusers = pd.read_csv(self.raw_file_white_opusers, dtype='str')
        all=[user_fraud , user_change, user_non_fraud,user_fraud_opusers,user_change_opusers,user_non_fraud_opusers]
        length=[len(x) for x in all]
        for i in range(1,len(length)):
            length[i]=length[i]+length[i-1]
        # 在行方向上拼接数据
        df = pd.concat(all, ignore_index=True)
        df=self.get_user_embed(df)
        return df[:length[0]],df[length[0]:length[1]],df[length[1]:length[2]],df[length[2]:length[3]],df[length[3]:length[4]],df[length[4]:length[5]],df[:length[2]]
    def get_user_embed(self,original_df):
        #对给定的属性列获取独热编码并存储在'hot'
        encoded_df = pd.get_dummies(original_df, columns=['开户接入方式','宽带标识'], prefix=['开户接入方式_dummies','宽带标识_dummies'])
        original_df=original_df[['开户接入方式','宽带标识']].join(encoded_df)
        z_list=['年龄','本月消费','本月使用流量/M','本月通话时长','3个月主叫占比','3个月主叫离散','3个月主叫外省号码占比']
        for name in z_list:
            original_df[name].fillna(0, inplace=True)
        return original_df
    def extract_hours(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[:2]
    # 自定义函数：根据字符串长度提取不同位置的字符
    def extract_minutes(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[2:4]
    # 自定义函数：根据字符串长度提取不同位置的字符
    def extract_seconds(self,time_str):
        time_str=time_str.zfill(6)
        return time_str[4:]
    def cylindrical_to_cartesian(self, r, theta, h):
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = h
        return x, y, z
    def cacu_time(self,df):
        z = (datetime.datetime(int(df['year']), int(df['month']), int(df['day']))-self.start_date).days
        r = 1
        alltime=24*60*60
        now=int(df['hour'])*3600+int(df['minute'])*60+int(df['second'])
        theta = now/alltime*2*math.pi
        x, y, z = self.cylindrical_to_cartesian(r, theta, z)
        return x,y,z
    # 获取通话时间特征
    def get_voc_feat(self,df):
        print('正在处理时间列。。。')
        df["start_datetime"] = pd.to_datetime(df['日期'])
        df["year"] = df["start_datetime"].dt.year
        df["month"] = df["start_datetime"].dt.month
        df["day"] = df["start_datetime"].dt.day
        df['hour'] = df['时间'].apply(self.extract_hours).astype('int64')
        df['minute'] = df['时间'].apply(self.extract_minutes).astype('int64')
        df['second'] = df['时间'].apply(self.extract_seconds).astype('int64')
        print('--------------已添加时间列！-----------')
        return df
    # 读取原始数据，对voc数据进行编码
    def get_voc_embed(self,original_df):
        #计算地址变化情况,注意需要确保话单是按照时间顺序的
        # 初始化新的列
        original_df['漫游地和对端号码归属相同'] = '-1'
        original_df['imei变化'] = '-1'
        original_df['小区变化'] = '-1'
        original_df['基站变化'] = '-1'
        original_df['漫游地变化'] = '-1'
        original_df['x'] = '0'
        original_df['y'] = '0'
        original_df['z'] = '0'
        # 遍历数据并更新列
        print('正在遍历voc数据，并进行处理。。。')
        for i, row in tqdm(original_df.iterrows(), total=len(original_df)):
            # 对每行额外更新通话时长和坐标信息
            if original_df.at[i, '呼叫类型'] == '2':
                original_df.at[i, '通话时长'] = -int(original_df.at[i, '通话时长'])
            else:
                original_df.at[i, '通话时长'] = int(original_df.at[i, '通话时长'])
            original_df.at[i, 'x'],original_df.at[i, 'y'],original_df.at[i, 'z']=self.cacu_time(original_df.iloc[i])
            # 判断漫游地和对端号码归属是否相同,先保证值不为空
            if pd.isna(original_df.at[i, '漫游地']) or pd.isna(original_df.at[i, '对端号码归属']):
                pass
            elif original_df.at[i, '漫游地'] == original_df.at[i, '对端号码归属']:
                original_df.at[i, '漫游地和对端号码归属相同'] = '1'
            #必须要求是同一个电话号码才开始比较
            if i>0 and original_df.at[i, '号码'] == original_df.at[i - 1, '号码']:
                # 判断属性是否变化
                if pd.isna(original_df.at[i, 'imei']) or pd.isna(original_df.at[i - 1, 'imei']):
                    pass
                elif original_df.at[i, 'imei'] != original_df.at[i - 1, 'imei']:
                    original_df.at[i, 'imei变化'] = '1'
                if pd.isna(original_df.at[i, '小区']) or pd.isna(original_df.at[i - 1, '小区']):
                    pass
                elif original_df.at[i, '小区'] != original_df.at[i - 1, '小区']:
                    original_df.at[i, '小区变化'] = '1'
                if pd.isna(original_df.at[i, '基站']) or pd.isna(original_df.at[i - 1, '基站']):
                    pass
                elif original_df.at[i, '基站'] != original_df.at[i - 1, '基站']:
                    original_df.at[i, '基站变化'] = '1'
                if pd.isna(original_df.at[i, '漫游地']) or pd.isna(original_df.at[i - 1, '漫游地']):
                    pass
                elif original_df.at[i, '漫游地'] != original_df.at[i - 1, '漫游地']:
                    original_df.at[i, '漫游地变化'] = '1'

        #对给定的属性列获取独热编码并存储在'dummies'列
        encoded_df = pd.get_dummies(original_df, columns=['呼叫类型','漫游地和对端号码归属相同','imei变化','小区变化','基站变化','漫游地变化'], prefix=['呼叫类型_dummies','漫游地和对端号码归属相同_dummies','imei变化_dummies','小区变化_dummies','基站变化_dummies','漫游地变化_dummies'])
        original_df=original_df[['呼叫类型','漫游地和对端号码归属相同','imei变化','小区变化','基站变化','漫游地变化']].join(encoded_df)
        z_list=['通话时长']
        for name in z_list:
            original_df[name].fillna(0, inplace=True)
        height=2
        original_df['z'] = (original_df['z'] - original_df['z'].min()) / (original_df['z'].max() - original_df['z'].min()) * height
        return original_df
    def get_and_transfer(self,data_path):
        len_fraud=self.voc_fraud.shape[0]
        len_change=self.voc_change.shape[0]
        len_non_fraud=self.voc_non_fraud.shape[0]
        # 在行方向上拼接数据
        df = pd.concat([self.voc_fraud, self.voc_change, self.voc_non_fraud], ignore_index=True)
        # 获取通话时间信息 这步骤就是把通话的时间拆分成了小时，分钟等等单独的信息保存在了通话信息表中
        df = self.get_voc_feat(df)
        return df[:len_fraud],df[len_fraud:len_fraud+len_change],df[len_fraud+len_change:],df
    def get_data_obj(self,data_path):
        #目的是获取带有label的voc信息，把内部数据集的voc数据格式转化为公开数据集格式(额外多一行label信息)，然后直接复用代码
        voc_fraud,voc_change,voc_non_fraud,voc_ori=self.get_and_transfer(data_path)
        #分别获得用户信息，并进行编码
        user_fraud,user_change,user_non_fraud,user_fraud_op,user_change_op,user_non_fraud_op,user_ori=self.get_users(data_path)
        fraud_name=voc_fraud['号码'].unique()
        change_name=voc_change['号码'].unique()
        non_fraud_name=voc_non_fraud['号码'].unique()

        #对通话voc记录进行保存，保存成图需要的Data对象
        all={}
        print('正在处理graph信息获得obj对象')
        fraud_obj=self.get_obj(voc_fraud,user_fraud,user_fraud_op,fraud_name,'fraud_')
        change_obj=self.get_obj(voc_change,user_change,user_change_op,change_name,'change_')
        #dic obj,需要用名字作为key以便后续存储
        non_fraud_obj=self.get_obj(voc_non_fraud,user_non_fraud,user_non_fraud_op,non_fraud_name,'non_fraud_')
        all.update(fraud_obj)
        all.update(change_obj)
        all.update(non_fraud_obj)
        return fraud_obj,change_obj,non_fraud_obj,all
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    def process(self):
        idx = 0
        exist_files=os.listdir(self.processed_dir+'_dataset9/')
        # 检查列表a是否全部包含在file_names中
        all_included = all(file_name in exist_files for file_name in self.all_processed_files)
        if not all_included:
            fraud_obj,change_obj,non_fraud_obj,all_obj=self.get_data_obj(self.raw_file_users)
            print('共有'+str(len(fraud_obj))+'个诈骗对象，正在输出')
            print('共有'+str(len(change_obj))+'个突变诈骗对象，正在输出')
            print('共有'+str(len(non_fraud_obj))+'个正常对象，正在输出')
            for name in all_obj.keys():
                data = all_obj[name]

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir+'_dataset9/', name))
                idx += 1
        else:
            print('文件已经都处理好，无需进一步处理')
    def change_mode(self,mode):
        self.mode=mode
    def len(self):
        if self.mode=='train':
            return len(self.train_name)
        elif self.mode=='val':
            return len(self.val_name)
        elif self.mode=='test':
            return len(self.test_name)
        elif self.mode=='change':
            return len(self.test_change_name)
        elif self.mode=='fraud':
            return len(self.test_fraud_name)
    def get(self, idx):
        #需要进行一些数据的维度翻转（适应类定义）和存储名字更改（dataloader的batch会根据x识别，所以不能直接存储）
        if self.mode=='train':
            data = torch.load(osp.join(self.processed_dir+'_dataset9/', self.train_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.voc_attr=data.voc_attr.unsqueeze(0).unsqueeze(0)
        elif self.mode=='val':
            data = torch.load(osp.join(self.processed_dir+'_dataset9/', self.val_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.voc_attr=data.voc_attr.unsqueeze(0).unsqueeze(0)
        elif self.mode=='test':
            data = torch.load(osp.join(self.processed_dir+'_dataset9/', self.test_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.voc_attr=data.voc_attr.unsqueeze(0).unsqueeze(0)
        elif self.mode=='change':
            data = torch.load(osp.join(self.processed_dir+'_dataset9/', self.test_change_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.voc_attr=data.voc_attr.unsqueeze(0).unsqueeze(0)
        elif self.mode=='fraud':
            data = torch.load(osp.join(self.processed_dir+'_dataset9/', self.test_fraud_name[idx]))
            data.x_temp=data.x
            data.x=None
            data.voc_attr=data.voc_attr.unsqueeze(0).unsqueeze(0)
        
        return data