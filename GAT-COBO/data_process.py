import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import scipy.sparse as spp
from sklearn.preprocessing import StandardScaler
from dgl.data.utils import save_graphs, save_info
import pickle
class ShaanxiDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='shaanxi')

    def normalize(self,mx):
        # Row-normalize sparse matrix
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = spp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    def remove_duplicate_edges(self, data):
        """
        去除字典中 'edges' 键对应的列表中重复的号码对，
        两个号码顺序不影响重复判断。

        参数:
        - data: 包含 'edges' 键的字典，'edges' 键对应的值是一个列表，包含多个号码对

        返回:
        - 去除重复号码对后的字典
        """
        unique_edges = []
        seen = set()

        for edge in data["edges"]:
            # 使用 frozenset 去掉顺序
            edge_set = frozenset(edge)
            if edge_set not in seen:
                unique_edges.append(edge)
                seen.add(edge_set)

        # 更新字典中的 'edges' 键
        data["edges"] = unique_edges

        return data
    def convert_edges_to_indices(self, data, name_dic, names):
        """
        将 edges 列表中的号码转换为对应的索引。
        
        参数:
        - data: 包含 'edges' 键的字典，'edges' 键对应的值是一个列表，包含多个号码对
        - names: 一个包含号码的列表
        
        返回:
        - 更新后的字典，其中 'edges' 列表中的每个号码都被转换为对应的索引
        """
        # 创建一个号码到索引的映射
        name_to_index = {name_dic[name]: idx for idx, name in enumerate(names)}

        # 转换 edges 中的每个号码为对应的索引
        for edge in data["edges"]:
            edge[0] = name_to_index[edge[0]]
            edge[1] = name_to_index[edge[1]]

        return data['edges']
    def create_adjacency_matrix(self, edges, num_nodes):
        """
        将边列表转换为稀疏邻接矩阵的 COO 格式。
        
        参数:
        - edges: 转换后的边的索引列表
        - num_nodes: 图中节点的数量
        
        返回:
        - 稀疏邻接矩阵（COO 格式）
        """
        rows = []
        cols = []
        data = []

        # 遍历每个边，将其转化为 COO 格式的三个数组
        for edge in edges:
            rows.append(edge[0])
            cols.append(edge[1])
            data.append(1)  # 假设每条边的权重为 1

        # 创建 COO 格式的邻接矩阵
        adj_matrix = spp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        return adj_matrix
    def process(self,path="./data/"):
        # load raw feature and labels
        #idx_features_labels = np.genfromtxt("{}{}.csv".format(path, "all_feat_with_label"),
        #                                    dtype=np.dtype(str), delimiter=',', skip_header=1)
        #features = spp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        #features = np.array(features)
        #normalize the feature with z-score
        #features=StandardScaler().fit_transform(np.asarray(features.todense()))
        #labels = np.array(idx_features_labels[:, -1], dtype=np.int_)
        import json
        with open('data/data_dic.pkl', 'rb') as pkl_file:
            restored_object = pickle.load(pkl_file)
        with open('data/name_dic.json') as f:
            name_dic = json.load(f)
        with open('data/edges_dic.json') as f:
            edges_dic = json.load(f)
        edges_dic = self.remove_duplicate_edges(edges_dic)
        
        features = np.concatenate((restored_object['X_train'],restored_object['X_val'],restored_object['X_test']),axis=0)
        labels = np.concatenate((restored_object['y_train'],restored_object['y_val'],restored_object['y_test']),axis=0)
        names = restored_object['name_train']+restored_object['name_val']+restored_object['name_test']
        fraud_names = restored_object['name_fraud']
        change_names = restored_object['name_change']
        adj = self.create_adjacency_matrix(self.convert_edges_to_indices(edges_dic, name_dic, names),9874)

        self.labels=torch.tensor(labels)
        node_features = torch.from_numpy(np.array(features))
        node_labels = torch.from_numpy(labels)

        adj = spp.coo_matrix(adj)
        # build symmetric adjacency matrix and normalize
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + spp.eye(9874))
        #adj = self.normalize(spp.eye(9874))

        self.graph = dgl.from_scipy(adj)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        # specify the default train,valid,test set for DGLgraph
        n_nodes = features.shape[0]
        n_train = int(5923)
        n_val = int(1973)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        fraud_mask = torch.zeros(n_nodes, dtype=torch.bool)
        mutation_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        i=0
        for n in names:
            if n in fraud_names:
                fraud_mask[i] = True
            if n in change_names:
                mutation_mask[i] = True
            i+=1
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.ndata['fraud_mask'] = fraud_mask
        self.graph.ndata['mutation_mask'] = mutation_mask
        num = 0
        for a in test_mask:
            if a == True:
                num+=1
        print(num)
        self.graph.num_labels = 2
        self._num_classes=2

        save_graphs('./data/Shaanxi_tele.bin', self.graph, {'labels': self.labels})
        save_info('./data/Shaanxi_tele.pkl', {'num_classes': self.num_classes})
        print('The Shaanxi dataset is successfully generated! ')

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes

class TelcomFraudDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='telcom_fraud')

    def normalize(self,mx):
        # Row-normalize sparse matrix
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = spp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    def read_txt(self, file_name):
        # 打开文件并读取每行
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # 去除每行的换行符并转化为列表
        lines = [line.strip() for line in lines]
        return lines
    def process(self,path="./data/"):
        # load raw feature and labels
        idx_features_labels = np.genfromtxt("{}{}.csv".format(path, "all_feat_with_label"),
                                            dtype=np.dtype(str), delimiter=',', skip_header=1)
        features = spp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        import pandas as pd

        # 读取 CSV 文件
        df = pd.read_csv('data/all_feat_with_label.csv')
        # 提取 'phone_no_m' 列并转换为列表
        phone_no_list = df['phone_no_m'].tolist()
        train_phones = self.read_txt('data/0.6name_train.txt')
        val_phones = self.read_txt('data/0.6name_val.txt')
        test_phones = self.read_txt('data/0.6name_test.txt')

        #features = np.array(features)
        #normalize the feature with z-score
        features=StandardScaler().fit_transform(np.asarray(features.todense()))
        labels = np.array(idx_features_labels[:, -1], dtype=np.int_)
        self.labels=torch.tensor(labels)
        node_features = torch.from_numpy(np.array(features))
        node_labels = torch.from_numpy(labels)

        # load adjacency matrix
        adj = spp.load_npz(path + 'node_adj_sparse.npz')
        adj = adj.toarray()
        adj = spp.coo_matrix(adj)
        # build symmetric adjacency matrix and normalize
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + spp.eye(adj.shape[0]))

        self.graph = dgl.from_scipy(adj)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        # specify the default train,valid,test set for DGLgraph
        n_nodes = features.shape[0]
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        i=0
        num_train =0
        num_val =0
        num_test =0
        for n in phone_no_list:
            if n in train_phones:
                train_mask[i] = True
                num_train+=1
            if n in val_phones:
                val_mask[i] = True
                num_val+=1
            if n in test_phones:
                test_mask[i] = True
                num_test+=1
            i+=1
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.num_labels = 2
        self._num_classes=2

        save_graphs('./data/Sichuan_tele.bin', self.graph, {'labels': self.labels})
        save_info('./data/Sichuan_tele.pkl', {'num_classes': self.num_classes})
        print('The Sichuan dataset is successfully generated! ',num_train,num_val,num_test)

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes


if __name__=="__main__":
    # process Sichuan dataset
    dataset = TelcomFraudDataset()
    # process shannxi dataset
    ShaanxiDataset()