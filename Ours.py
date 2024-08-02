import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid,ModelNet
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, PointNetConv, fps, radius
from torch_geometric.nn import global_max_pool
import sys
sys.path.append("/xxx/xxx/Mutation_Experiment/")
from src.models.GNNs.Shannxi_Datasets import Graph_Dataset2
from torch.optim import lr_scheduler
import random
from glob import glob
from tqdm.auto import tqdm
import wandb
import os
import os.path as osp
import argparse
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.nn import BatchNorm1d
WANDB_API_KEY='xxx'
WANDB_CACHE_DIR='./cache'
WANDB_CONFIG_DIR='./config'
WANDB_DIR='./wandb'
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
os.environ['WANDB_CACHE_DIR'] = WANDB_CACHE_DIR
os.environ['WANDB_CONFIG_DIR'] = WANDB_CONFIG_DIR
os.environ['WANDB_DIR'] = WANDB_DIR
#os.environ['WANDB_MODE'] = 'dryrun'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
np.random.seed(0)
random_seed = 0
best=0
best_test=0
cuda_device=1

# 创建一个GAT模型
class GATModel(nn.Module):
    def __init__(self, in_channels, hide_channels, out_channels, num_heads, dropout=0.6):
        super(GATModel, self).__init__()
        print(hide_channels)
        self.conv1 = GATConv(in_channels, hide_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hide_channels*num_heads, out_channels, heads=num_heads, dropout=dropout)
    def get_true_index(self, edge_index, edge_attr):
        change_positions=[0]
        x_tensors=[]
        y_tensors=[]
        lengths=[]
        edges=[]
        now1=0
        now2=1
        edges.append([now1,now2])
        for i in range(1,len(edge_index[0])):
            #去掉自回路边
            if edge_index[0,i] == edge_index[1,i]:
                edge_attr = torch.cat((edge_attr[:i], edge_attr[i + 1:]), dim=0)
                continue
            if edge_index[0,i] != edge_index[0,i-1]:
                now1=now2+1
                now2=now1+1
                edges.append([now1,now2])
            else:
                now2+=1
                edges.append([now1,now2])
        edge_index=torch.tensor(edges).permute(1, 0).to(edge_index.device)
        return edge_index, edge_attr
    def forward(self, data):
        x, edge_index, edge_attr = data.x_temp.type(torch.float32), data.edge_index, data.edge_attr.type(torch.float32)
        #dataloader 返回的index是错误的，需要重新修改
        edge_index, edge_attr=self.get_true_index(edge_index, edge_attr)
        root_index= torch.nonzero(data.root).squeeze()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x[root_index].reshape(-1,x.shape[-1])

class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, ball_query_radius, nn):
        super().__init__()
        self.ratio = ratio
        self.ball_query_radius = ball_query_radius
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.ball_query_radius,
            batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch
class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(
            x.size(0),
            device=batch.device
        )
        return x, pos, batch  
class PointNetPlusPlus(torch.nn.Module):
    def __init__(
        self,
        set_abstraction_ratio_1, set_abstraction_ratio_2,
        set_abstraction_radius_1, set_abstraction_radius_2, dropout
    ):
        super().__init__()
        self.sa1_module = SetAbstraction(set_abstraction_ratio_1, set_abstraction_radius_1, MLP([16+3, 64, 64, 128]))
        self.sa2_module = SetAbstraction(set_abstraction_ratio_2, set_abstraction_radius_2, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSetAbstraction(MLP([256 + 3, 256, 512, 1024]))
        # 添加第一个 BN 层
        self.bn1 = BatchNorm1d(16)

    def forward(self, data):
        input = self.bn1(data.voc_attr)
        sa0_out = (input, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return x
class LSTM1D(nn.Module):
    def __init__(self, input_dim, hidden_dim1):
        super(LSTM1D, self).__init__()
        
        self.lstm1d = nn.LSTM(input_size=input_dim,
                                hidden_size=hidden_dim1,
                                num_layers=2,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)
        # 添加第一个 BN 层
        self.bn1 = BatchNorm1d(input_dim)

    def forward(self, x):
        # 使用 BN1
        x = self.bn1(x)
        x = x.permute(0,2,1)
        x = self.lstm1d(x)
        return x
        
class TCCNN(torch.nn.Module):
    def __init__(
        self,
        set_abstraction_ratio_1, set_abstraction_ratio_2,
        set_abstraction_radius_1, set_abstraction_radius_2, dropout,
        num_features, hide_features, out_features, num_heads
    ):
        super().__init__()
        self.point_net = PointNetPlusPlus(set_abstraction_ratio_1, set_abstraction_ratio_2, set_abstraction_radius_1, set_abstraction_radius_2, dropout)
        self.gat = GATModel(num_features, hide_features, out_features, num_heads, dropout=dropout)
        input_dim = 16  # 输入特征的维度
        hidden_dim1 = 32
        
        self.cnn_net1 = LSTM1D(input_dim,hidden_dim1)
        self.mlp = MLP([1024+out_features*num_heads+2*hidden_dim1, 512, 256, 2], dropout=dropout, norm=None)
        
    def get_input(self, voc, batch):
        start=[0]
        voc_new=[]
        for i in range(1,len(batch)):
            if batch[i]!=batch[i-1]:
                start.append(i)
        for i in range(len(start)-1):
            voc_new.append(voc[start[i]:start[i+1]])
        voc_new.append(voc[start[-1]:])
        # 找到最大的第一维长度,开始填充为3dtensor
        max_len = max(tensor.size(0) for tensor in voc_new)
        feature_dim=voc_new[0].size(1)
        # 创建一个零填充的三维张量
        padded_tensor = torch.zeros(len(voc_new), max_len, feature_dim)
        # 将二维张量填充到三维张量中
        for i, tensor in enumerate(voc_new):
            padded_tensor[i, :tensor.size(0), :tensor.size(1)] = tensor
        
        padded_tensor = padded_tensor.permute(0,2,1)
        return padded_tensor.to(device)
    def forward(self, data):
        point_feature=self.point_net(data)
        gat_feature=self.gat(data)
        new_data=self.get_input(data.voc_attr,data.batch)
        cnn_feature, (hn, cn)=self.cnn_net1(new_data)
        cnn_feature, max_index =torch.max(cnn_feature, dim=-2)
        cnn_feature=cnn_feature.view(-1,cnn_feature.shape[-1])
        feature=torch.cat([point_feature,cnn_feature,gat_feature],dim=-1)
        return self.mlp(feature).log_softmax(dim=-1)
    
def train_step(epoch):
    """Training Step"""
    model.train()
    epoch_loss, correct = 0, 0
    num_train_examples = len(train_loader)
    global best
    global best_test
    progress_bar = tqdm(
        range(num_train_examples),
        desc=f"Training Epoch {epoch}/{config.epochs}"
    )
    a=iter(train_loader)
    for batch_idx in progress_bar:
        data = next(a).to(device)
        #print(data)
        optimizer.zero_grad()
        prediction = model(data)
        loss = F.nll_loss(prediction, data.y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        correct += prediction.max(1)[1].eq(data.y).sum().item()
        
        if batch_idx!=0 and batch_idx%100==0:
            dic={'epoch':epoch+batch_idx/num_train_examples}
            val_dic=val_step(epoch+batch_idx/num_train_examples)
            dic.update(val_dic)
            test_dic=test_step(epoch+batch_idx/num_train_examples)
            dic.update(test_dic)

            dic.update(change_step(epoch+batch_idx/num_train_examples))
            dic.update(fruad_step(epoch+batch_idx/num_train_examples))
            if val_dic['Validation/'+args.best_metric]>best:
                best=val_dic['Validation/'+args.best_metric]
                best_test=dic

            wandb.log(dic)
            model.train()
    
    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / len(train_loader.dataset)
    
    wandb.log({
        'epoch':epoch+1,
        "Train/Loss": epoch_loss,
        "Train/Accuracy": epoch_accuracy
    })

def val_step(epoch):
    """Validation Step"""
    model.eval()
    epoch_loss, correct = 0, 0
    num_val_examples = len(val_loader)
    pre=[]
    tr=[]
    pro=[]
    progress_bar = tqdm(
        range(num_val_examples),
        desc=f"Validation Epoch {epoch}/{config.epochs}"
    )
    a=iter(val_loader)
    for batch_idx in progress_bar:
        data = next(a).to(device)
        with torch.no_grad():
            prediction = model(data)
        
        loss = F.nll_loss(prediction, data.y)
        epoch_loss += loss.item()

        pro.append(prediction[:, 1])
        predicted_labels = torch.argmax(prediction, dim=1)
        pre.append(predicted_labels)
        true_labels = data.y
        tr.append(true_labels)
    pre=torch.cat(pre,dim=0).cpu().numpy()
    tr=torch.cat(tr,dim=0).cpu().numpy()
    pro=torch.cat(pro,dim=0).cpu().numpy()
    epoch_accuracy = accuracy_score(tr,pre)
    precision, recall, f1, _ = precision_recall_fscore_support(tr, pre, average="macro")
    auc = roc_auc_score(tr, pro)

    epoch_loss = epoch_loss / num_val_examples
    
    print('val_results')
    print(epoch_loss,epoch_accuracy,f1,auc)
    
    return {
        "Validation/Loss": epoch_loss,
        "Validation/Accuracy": epoch_accuracy,
        "Validation/precision":precision, 
        "Validation/recall":recall,
        "Validation/f1": f1,
        "Validation/auc": auc
    }
def test_step(epoch):
    """test Step"""
    model.eval()
    epoch_loss, correct = 0, 0
    num_test_examples = len(test_loader)
    pre=[]
    tr=[]
    pro=[]
    progress_bar = tqdm(
        range(num_test_examples),
        desc=f"Test Epoch {epoch}/{config.epochs}"
    )
    a=iter(test_loader)
    for batch_idx in progress_bar:
        data = next(a).to(device)
        with torch.no_grad():
            prediction = model(data)
        
        loss = F.nll_loss(prediction, data.y)
        epoch_loss += loss.item()

        pro.append(prediction[:, 1])
        predicted_labels = torch.argmax(prediction, dim=1)
        pre.append(predicted_labels)
        true_labels = data.y
        tr.append(true_labels)
    pre=torch.cat(pre,dim=0).cpu().numpy()
    tr=torch.cat(tr,dim=0).cpu().numpy()
    pro=torch.cat(pro,dim=0).cpu().numpy()
    epoch_accuracy = accuracy_score(tr,pre)
    precision, recall, f1, _ = precision_recall_fscore_support(tr, pre, average="macro")
    auc = roc_auc_score(tr, pro)

    epoch_loss = epoch_loss / num_test_examples
    
    print('test_results')
    print(epoch_loss,epoch_accuracy,f1,auc)
    
    return {
        "Test/Loss": epoch_loss,
        "Test/Accuracy": epoch_accuracy,
        "Test/precision":precision, 
        "Test/recall":recall,
        "Test/f1": f1,
        "Test/auc": auc
    }

def change_step(epoch):
    """change_Validation Step"""
    model.eval()
    epoch_loss, correct = 0, 0
    num_val_examples = len(change_loader)
    pre=[]
    tr=[]
    pro=[]
    progress_bar = tqdm(
        range(num_val_examples),
        desc=f"change_Validation Epoch {epoch}/{config.epochs}"
    )
    a=iter(change_loader)
    for batch_idx in progress_bar:
        data = next(a).to(device)
        with torch.no_grad():
            prediction = model(data)
        
        loss = F.nll_loss(prediction, data.y)
        epoch_loss += loss.item()

        pro.append(prediction[:, 1])
        predicted_labels = torch.argmax(prediction, dim=1)
        pre.append(predicted_labels)
        true_labels = data.y
        tr.append(true_labels)
    pre=torch.cat(pre,dim=0).cpu().numpy()
    tr=torch.cat(tr,dim=0).cpu().numpy()
    pro=torch.cat(pro,dim=0).cpu().numpy()
    epoch_accuracy = accuracy_score(tr,pre)

    epoch_loss = epoch_loss / num_val_examples
    
    print('change_val_results')
    print(epoch_loss,epoch_accuracy)
    
    return {
        "change_Validation/Loss": epoch_loss,
        "change_Validation/Accuracy": epoch_accuracy,
    }

def fruad_step(epoch):
    """fraud_Validation Step"""
    model.eval()
    epoch_loss, correct = 0, 0
    num_val_examples = len(fraud_loader)
    pre=[]
    tr=[]
    pro=[]
    progress_bar = tqdm(
        range(num_val_examples),
        desc=f"fraud_Validation Epoch {epoch}/{config.epochs}"
    )
    a=iter(fraud_loader)
    for batch_idx in progress_bar:
        data = next(a).to(device)
        with torch.no_grad():
            prediction = model(data)
        
        loss = F.nll_loss(prediction, data.y)
        epoch_loss += loss.item()

        pro.append(prediction[:, 1])
        predicted_labels = torch.argmax(prediction, dim=1)
        pre.append(predicted_labels)
        true_labels = data.y
        tr.append(true_labels)
    pre=torch.cat(pre,dim=0).cpu().numpy()
    tr=torch.cat(tr,dim=0).cpu().numpy()
    pro=torch.cat(pro,dim=0).cpu().numpy()
    epoch_accuracy = accuracy_score(tr,pre)

    epoch_loss = epoch_loss / num_val_examples
    
    print('fraud_val_results')
    print(epoch_loss,epoch_accuracy)
    
    return {
        "fraud_Validation/Loss": epoch_loss,
        "fraud_Validation/Accuracy": epoch_accuracy,
    }

def visualize_evaluation(table, epoch):
    """Visualize validation result in a Weights & Biases Table"""
    point_clouds, losses, predictions, ground_truths, is_correct = [], [], [], [], []
    progress_bar = tqdm(
        range(config.num_visualization_samples),
        desc=f"Generating Visualizations for Epoch {epoch}/{config.epochs}"
    )
    
    for idx in progress_bar:
        data = next(iter(vizualization_loader)).to(device)
        
        with torch.no_grad():
            prediction = model(data)
        
        point_clouds.append(
            wandb.Object3D(torch.squeeze(data.pos, dim=0).cpu().numpy())
        )
        losses.append(F.nll_loss(prediction, data.y).item())
        predictions.append(config.categories[int(prediction.max(1)[1].item())])
        ground_truths.append(config.categories[int(data.y.item())])
        is_correct.append(prediction.max(1)[1].eq(data.y).sum().item())
    
    table.add_data(
        epoch, point_clouds, losses, predictions, ground_truths, is_correct
    )
    return table


def save_checkpoint(epoch):
    """Save model checkpoints as Weights & Biases artifacts"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, "checkpoint.pt")
    
    artifact_name = wandb.util.make_artifact_name_safe(
        f"{wandb.run.name}-{wandb.run.id}-checkpoint"
    )
    
    checkpoint_artifact = wandb.Artifact(artifact_name, type="checkpoint")
    checkpoint_artifact.add_file("checkpoint.pt")
    wandb.log_artifact(
        checkpoint_artifact, aliases=["latest", f"epoch-{epoch}"]
    )
def save_csv(data):
    import csv
    # CSV文件路径
    csv_file_path = "results.csv"

    # 将字典写入CSV文件
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = data.keys()

        # 创建CSV写入对象
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 如果文件为空，写入列名
        if csvfile.tell() == 0:
            writer.writeheader()

        # 写入字典数据
        writer.writerow(data)

    print("数据已成功写入CSV文件。")

parser = argparse.ArgumentParser(description='一个带有命令行参数的示例脚本')
    
# 添加命令行参数
parser.add_argument('--wandb_project', type=str, default="shannxi_paper_results")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--hide_features', type=int, default=16)
parser.add_argument('--out_features', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
parser.add_argument('--best_metric', type=str, default='auc', help='选择根据哪个metric保存验证集最优模型')
parser.add_argument('--device', type=int, default=0, help='选择训练的gpu')
args = parser.parse_args()
model_name="experiment/ours"
args.wandb_run_name=model_name+'_bs_'+str(args.batch_size)+'_epo_'+str(args.epochs)+'_lr_'+str(args.learning_rate)+'_hf_'+str(args.hide_features)+'_of_'+str(args.out_features)
cuda_device=args.device
#  点云代码！！！！！！！！！！！！！
wandb_project = args.wandb_project #@param {"type": "string"}
wandb_run_name = args.wandb_run_name #@param {"type": "string"}

wandb.init(project=wandb_project, name=wandb_run_name, job_type="baseline-train")

# # Set experiment configs to be synced with wandb
config = wandb.config
#config.modelnet_dataset_alias = "ModelNet10" #@param ["ModelNet10", "ModelNet40"] {type:"raw"}

config.seed = 4242 #@param {type:"number"}
random.seed(config.seed)
torch.manual_seed(config.seed)

config.sample_points = 2048 #@param {type:"slider", min:256, max:4096, step:16}

config.categories = ['non_fruad','fruad']

config.batch_size = args.batch_size #@param {type:"slider", min:4, max:128, step:4}
config.num_workers = 6 #@param {type:"slider", min:1, max:10, step:1}

config.device = torch.device('cuda',cuda_device)
device = torch.device(config.device)

config.set_abstraction_ratio_1 = 0.748 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
config.set_abstraction_radius_1 = 0.4817 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
config.set_abstraction_ratio_2 = 0.3316 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
config.set_abstraction_radius_2 = 0.2447 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
config.dropout = 0.1 #@param {type:"slider", min:0.1, max:1.0, step:0.1}
# # 定义gat超参数
config.num_features = 23
config.hide_features = args.hide_features
config.out_features = args.out_features
config.num_heads = 8


config.learning_rate = args.learning_rate #@param {type:"number"}
config.epochs = args.epochs #@param {type:"slider", min:1, max:100, step:1}
config.num_visualization_samples = 20 #@param {type:"slider", min:1, max:100, step:1}

pre_transform = T.NormalizeScale()
transform = T.SamplePoints(config.sample_points)

train_dataset = Graph_Dataset2(
    root='Shannxi_Graph/',
    mode='train',
    pre_filter=None,
    pre_transform=None
)
train_dataset.get_tranformer()

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers
)

val_dataset = Graph_Dataset2(
    root='Shannxi_Graph/',
    mode='val',
    pre_filter=None,
    pre_transform=None
)
val_dataset.get_tranformer()

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)
test_dataset = Graph_Dataset2(
    root='Shannxi_Graph/',
    mode='test',
    pre_filter=None,
    pre_transform=None
)
test_dataset.get_tranformer()
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)

change_dataset = Graph_Dataset2(
    root='Shannxi_Graph/',
    mode='change',
    pre_filter=None,
    pre_transform=None
)
change_dataset.get_tranformer()

change_loader = DataLoader(
    change_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)

fraud_dataset = Graph_Dataset2(
    root='Shannxi_Graph/',
    mode='fraud',
    pre_filter=None,
    pre_transform=None
)
fraud_dataset.get_tranformer()

fraud_loader = DataLoader(
    fraud_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)

random_indices = random.sample(
    list(range(len(val_dataset))),
    config.num_visualization_samples
)
vizualization_loader = DataLoader(
    [val_dataset[idx] for idx in random_indices],
    batch_size=1,
    shuffle=False,
    num_workers=config.num_workers
)

# Define TCCNN model.
model = TCCNN(
    config.set_abstraction_ratio_1,
    config.set_abstraction_ratio_2,
    config.set_abstraction_radius_1,
    config.set_abstraction_radius_2,
    config.dropout,
    config.num_features,
    config.hide_features,
    config.out_features,
    config.num_heads
)

model.to(device)
# Define Optimizer
# 定义不同模块的学习率
point_net_lr = config.learning_rate  # 设置point_net模块的学习率
gat_lr = config.learning_rate*10       # 设置gat模块的学习率
mlp_lr = config.learning_rate      # 设置mlp模块的学习率
lstm_lr = config.learning_rate       # 设置lstm模块的学习率
# 将模型的参数按照模块分组
point_net_params = list(model.point_net.parameters())
gat_params = list(model.gat.parameters())
mlp_params = list(model.mlp.parameters())
lstm_net_params = list(model.cnn_net1.parameters())
# 创建优化器，并将不同模块的参数添加到不同的参数组
optimizer = optim.Adam([
    {'params': point_net_params, 'lr': point_net_lr},
    {'params': gat_params, 'lr': gat_lr},
    {'params': lstm_net_params, 'lr': lstm_lr},
    {'params': mlp_params, 'lr': mlp_lr}
])
table = wandb.Table(
    columns=[
        "Epoch",
        "Point-Clouds",
        "Losses",
        "Predicted-Classes",
        "Ground-Truth",
        "Is-Correct"
    ]
)
for epoch in range(1, config.epochs + 1):
    train_step(epoch-1)
    dic={'epoch':epoch}
    val_dic=val_step(epoch)
    dic.update(val_dic)
    test_dic=test_step(epoch)
    dic.update(test_dic)
    dic.update(change_step(epoch))
    dic.update(fruad_step(epoch))
    wandb.log(dic)
    if val_dic['Validation/'+args.best_metric]>best:
        best=val_dic['Validation/'+args.best_metric]
        best_test=dic
wandb.log({"Evaluation": table})

wandb.finish()
#保存到本地最优数据
best_test['model_name']=model_name
save_csv(best_test)


