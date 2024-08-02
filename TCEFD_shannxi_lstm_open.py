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
sys.path.append("/home/baihaitao/CDR2IMG/")
from src.models.GNNs.Shannxi_Datasets import Graph_Dataset5
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
from sklearn.preprocessing import MinMaxScaler
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
cuda_device=5

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

        input_dim = 10  # 输入特征的维度
        hidden_dim1 = 32
        
        self.cnn_net1 = LSTM1D(input_dim,hidden_dim1)
        self.mlp = MLP([2*hidden_dim1, 512, 256, 2], dropout=dropout, norm=None)
        
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
        #point_feature=self.point_net(data)
        #gat_feature=self.gat(data)
        new_data=self.get_input(data.voc_attr,data.batch)
        cnn_feature, (hn, cn)=self.cnn_net1(new_data)
        cnn_feature, max_index =torch.max(cnn_feature, dim=-2)
        cnn_feature=cnn_feature.view(-1,cnn_feature.shape[-1])
        feature=cnn_feature
        return self.mlp(feature).log_softmax(dim=-1)
    
def train_step(epoch):
    """Training Step"""
    model.train()
    epoch_loss, correct = 0, 0
    num_train_examples = len(train_loader)
    
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
        
        if batch_idx!=0 and batch_idx%50==0:
            dic={'epoch':epoch+batch_idx/num_train_examples}
            dic.update(val_step(epoch+batch_idx/num_train_examples))
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
    #f1 = f1_score(pre, tr, average='macro')
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

parser = argparse.ArgumentParser(description='一个带有命令行参数的示例脚本')
    
# 添加命令行参数
parser.add_argument('--wandb_project', type=str, default="dianxin_open")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--hide_features', type=int, default=16)
parser.add_argument('--out_features', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
args = parser.parse_args()
args.wandb_run_name="experiment/tccnn_lstm_open"+'_bs_'+str(args.batch_size)+'_epo_'+str(args.epochs)+'_lr_'+str(args.learning_rate)+'_hf_'+str(args.hide_features)+'_of_'+str(args.out_features)

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
#config.num_features = 33
config.num_features = 29
config.hide_features = args.hide_features
config.out_features = args.out_features
#config.out_features = 2
config.num_heads = 8


config.learning_rate = args.learning_rate #@param {type:"number"}
config.epochs = args.epochs #@param {type:"slider", min:1, max:100, step:1}
config.num_visualization_samples = 20 #@param {type:"slider", min:1, max:100, step:1}

pre_transform = T.NormalizeScale()
transform = T.SamplePoints(config.sample_points)

train_dataset = Graph_Dataset5(
    root='Shannxi_Graph/',
    mode='train_name_06',
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

val_dataset = Graph_Dataset5(
    root='Shannxi_Graph/',
    mode='test_name_06',
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

mlp_lr = config.learning_rate      # 设置mlp模块的学习率
lstm_lr = config.learning_rate*5       # 设置gat模块的学习率
# 将模型的参数按照模块分组
mlp_params = list(model.mlp.parameters())
lstm_net_params = list(model.cnn_net1.parameters())
# 创建优化器，并将不同模块的参数添加到不同的参数组
optimizer = optim.Adam([
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
    dic.update(val_step(epoch))
    wandb.log(dic)
    save_checkpoint(epoch)
wandb.log({"Evaluation": table})

wandb.finish()


