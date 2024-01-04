#!/bin/sh

train(){
    filename=$1
    project_name=$2
    batch_size=$3
    hide_features=$4
    out_features=$5
    epochs=$6
    learning_rate=$7
    python ${filename} --wandb_project ${project_name} --batch_size ${batch_size} --hide_features ${hide_features} --out_features ${out_features} --epochs ${epochs} --learning_rate ${learning_rate} --device 0 &&
    echo ${filename}
}

train TCEFD_shannxi_gcn_voc.py shannxi_paper_results 16 32 64 10 1e-4 &&
train TCEFD_shannxi_gnn_voc.py shannxi_paper_results 16 32 64 10 1e-4 &&
train TCEFD_shannxi_gnn_voc_week.py shannxi_paper_results 16 32 64 10 1e-4 &&
train TCEFD_shannxi_gnn_voc_day.py shannxi_paper_results 16 32 64 10 1e-4 &&
train Ours.py shannxi_paper_results 16 32 64 10 1e-4 &&
train TCEFD_shannxi_cdr2img.py shannxi_paper_results 16 32 64 10 1e-4 &
