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
from src.models.GNNs.Shannxi_Datasets import Graph_Dataset9
from torch.nn import BatchNorm1d
from torch.optim import lr_scheduler
import random
from glob import glob
from tqdm.auto import tqdm
import wandb
import os
import os.path as osp
import argparse
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

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


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(6,6),padding=(2,2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(6,6))
        self.pool = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(128, 64, kernel_size=(6,6),padding=(2,2))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(6,6))
        self.pool = nn.MaxPool2d((2,2))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6*128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        x=data.voc_attr.float()
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.softmax(x)

        return x

# 创建模型实例
model = CNNModel()

class TCCNN(torch.nn.Module):
    def __init__(
        self,
        set_abstraction_ratio_1, set_abstraction_ratio_2,
        set_abstraction_radius_1, set_abstraction_radius_2, dropout,
        num_features, hide_features, out_features, num_heads
    ):
        super().__init__()
        self.point_net = CNNModel()


    def forward(self, data):
        point_feature=self.point_net(data)

        return point_feature.log_softmax(dim=-1)
    
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
model_name="experiment/cdr2img"
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

config.num_features = 23
config.hide_features = args.hide_features
config.out_features = args.out_features

config.num_heads = 8


config.learning_rate = args.learning_rate #@param {type:"number"}
config.epochs = args.epochs #@param {type:"slider", min:1, max:100, step:1}
config.num_visualization_samples = 20 #@param {type:"slider", min:1, max:100, step:1}

pre_transform = T.NormalizeScale()
transform = T.SamplePoints(config.sample_points)

train_dataset = Graph_Dataset9(
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

val_dataset = Graph_Dataset9(
    root='Shannxi_Graph/',
    mode='val',
    pre_filter=None,
    pre_transform=None
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)
test_dataset = Graph_Dataset9(
    root='Shannxi_Graph/',
    mode='test',
    pre_filter=None,
    pre_transform=None
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)

change_dataset = Graph_Dataset9(
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

fraud_dataset = Graph_Dataset9(
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


# 将模型的参数按照模块分组
point_net_params = list(model.point_net.parameters())


# 创建优化器，并将不同模块的参数添加到不同的参数组
optimizer = optim.Adam([
    {'params': point_net_params, 'lr': point_net_lr},

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


