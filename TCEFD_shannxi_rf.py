import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid,ModelNet
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, PointNetConv, fps, radius
from torch_geometric.nn import global_max_pool
from torch.nn import BatchNorm1d
import sys
from sklearn.ensemble import RandomForestClassifier  # 用于分类问题
from sklearn.preprocessing import MinMaxScaler
sys.path.append("/home/baihaitao/CDR2IMG/")
from src.models.GNNs.Shannxi_Datasets import Graph_Dataset3
from torch.optim import lr_scheduler
import random
from glob import glob
from tqdm.auto import tqdm
import wandb
import os
import os.path as osp
import argparse
from sklearn.metrics import precision_score, recall_score,f1_score, roc_auc_score, accuracy_score

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

    
def train_step(epoch):
    """Training Step"""
    model.train()
    epoch_loss, correct = 0, 0
    num_train_examples = len(train_loader)
    criterion = nn.HingeEmbeddingLoss()
    progress_bar = tqdm(
        range(num_train_examples),
        desc=f"Training Epoch {epoch}/{config.epochs}"
    )
    a=iter(train_loader)
    for batch_idx in progress_bar:
        data = next(a).to(device)
        optimizer.zero_grad()
        
        prediction = model(data)
        print(prediction.shape)
        loss = criterion(prediction, data.y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        correct += prediction.max(1)[1].eq(data.y).sum().item()
        
        if batch_idx!=0 and batch_idx%100==0:
            dic={'epoch':epoch+batch_idx/num_train_examples}
            dic.update(val_step(epoch+batch_idx/num_train_examples))
            dic.update(change_step(epoch+batch_idx/num_train_examples))
            dic.update(fruad_step(epoch+batch_idx/num_train_examples))
            wandb.log(dic)
            model.train()
    
    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / len(train_loader.dataset)
    
    wandb.log({
        'epoch':epoch+1,
        "Train/Loss": epoch_loss,
        "Train/Accuracy": epoch_accuracy
    })

def val_step(epoch,X, y):
    """Validation Step"""
    tr=y
    pro=model.predict_proba(X)
    pre=np.argmax(pro, axis=1)
    epoch_accuracy = accuracy_score(tr,pre)
    f1 = f1_score(pre, tr, average='macro')
    precision=precision_score(pre, tr, average='macro')
    recall=recall_score(pre, tr, average='macro')

    auc = roc_auc_score(tr, pro[:,1])

    
    print('val_results')
    print(epoch_accuracy,f1,auc)
    
    return {
        "Test/Accuracy": epoch_accuracy,
        "Test/f1": f1,
        "Test/precision": precision,
        "Test/recall": recall,
        "Test/auc": auc
    }

def change_step(epoch,X, y):
    """change_Validation Step"""
    tr=y
    pro=model.predict_proba(X)
    pre=np.argmax(pro, axis=1)
    epoch_accuracy = accuracy_score(tr,pre)

    
    print('change_results')
    print(epoch_accuracy)
    
    return {
        "change_Validation/Accuracy": epoch_accuracy,
    }

def fruad_step(epoch,X, y):
    """fraud_Validation Step"""
    tr=y
    pro=model.predict_proba(X)
    pre=np.argmax(pro, axis=1)
    epoch_accuracy = accuracy_score(tr,pre)

    
    print('fruad_results')
    print(epoch_accuracy)

    return {
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

def get_svm_data(loader):
    epoch_loss, correct = 0, 0
    num_val_examples = len(loader)

    progress_bar = tqdm(
        range(num_val_examples),
        desc=f"get data"
    )
    a=iter(loader)
    all_x=[]
    all_y=[]
    #分配数据
    print('正在进行数据分配')
    for batch_idx in progress_bar:
        data = next(a).to(device)
        root_index= torch.nonzero(data.root).squeeze()
        feature=data.x_temp[root_index].reshape(-1,data.x_temp.shape[-1]).type(torch.float32)
        label=data.y
        all_x.append(feature)
        all_y.append(label)
    x=torch.cat(all_x,dim=0).cpu().numpy()
    y=torch.cat(all_y,dim=0).cpu().numpy()
    print(x.shape,y.shape)
    return x,y
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
args = parser.parse_args()
args.wandb_run_name="experiment/svm"+'_bs_'+str(args.batch_size)+'_epo_'+str(args.epochs)+'_lr_'+str(args.learning_rate)+'_hf_'+str(args.hide_features)+'_of_'+str(args.out_features)

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
config.num_features = 64
config.hide_features = args.hide_features
config.out_features = args.out_features
#config.out_features = 2
config.num_heads = 8


config.learning_rate = args.learning_rate #@param {type:"number"}
config.epochs = args.epochs #@param {type:"slider", min:1, max:100, step:1}
config.num_visualization_samples = 20 #@param {type:"slider", min:1, max:100, step:1}

pre_transform = T.NormalizeScale()
transform = T.SamplePoints(config.sample_points)

train_dataset = Graph_Dataset3(
    root='Shannxi_Graph/',
    mode='train',
    pre_filter=None,
    pre_transform=None
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers
)
X_train, y_train=get_svm_data(train_loader)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train) 
val_dataset = Graph_Dataset3(
    root='Shannxi_Graph/',
    mode='test',
    pre_filter=None,
    pre_transform=None
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)
X_val, y_val=get_svm_data(val_loader)
X_val = scaler.transform(X_val)
change_dataset = Graph_Dataset3(
    root='Shannxi_Graph/',
    mode='change',
    pre_filter=None,
    pre_transform=None
)

change_loader = DataLoader(
    change_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)
X_change, y_change=get_svm_data(change_loader)
X_change = scaler.transform(X_change)
fraud_dataset = Graph_Dataset3(
    root='Shannxi_Graph/',
    mode='fraud',
    pre_filter=None,
    pre_transform=None
)

fraud_loader = DataLoader(
    fraud_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)
X_fraud, y_fraud=get_svm_data(fraud_loader)
X_fraud = scaler.transform(X_fraud)
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
model = RandomForestClassifier(n_estimators=100)  # 设置 probability=True 以获取预测概率

model.fit(X_train, y_train)
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
epoch=10
dic={'epoch':epoch}
dic.update(val_step(epoch,X_val, y_val))
dic.update(change_step(epoch,X_change, y_change))
dic.update(fruad_step(epoch,X_fraud, y_fraud))
wandb.log(dic)


wandb.finish()
dic['model_name']='RandomForestClassifier'
save_csv(dic)


