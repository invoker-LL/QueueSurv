#!/bin/bash

gpuid=$1
task=$2
target_col=$3
split_dir=$4
split_names=$5
statistic=$6

feat='extracted-uni_ep20_baseline_NLL'
input_dim=1024
# input_dim=768
embed_dim=1024
mag='20x'
patch_size=224

model_tuple='ABMIL,default'
mag='20x'
max_epoch=20
bag_size='-1'
batch_size=1
lr=0.0001
wd=0.00001
loss_fn='nll'
alpha=0.5
opt='adamW'
lr_scheduler='cosine'
n_fc_layer=0
n_label_bin=4
es_flag=0
save_dir_root='results'

IFS=',' read -r model config_suffix <<< "${model_tuple}"

model_config=${model}_${config_suffix}
feat_name=$(echo $feat | sed 's/^extracted-//')
exp_code=${task}::${model_config}::${feat_name}
save_dir=${save_dir_root}/${exp_code}

th=0.00005
if awk "BEGIN {exit !($lr <= $th)}"; then
  warmup=0
  curr_lr_scheduler='constant'
else
  curr_lr_scheduler=$lr_scheduler
  warmup=1
fi

if [[ $bag_size == "-1" ]]; then
  curr_bag_size=$bag_size
  curr_batch_size=1
  grad_accum=32

  if [[ $loss_fn == 'cox' ]]; then
    alpha=0
  fi
else
  if [[ $loss_fn == 'cox' ]]; then
    continue
  fi

  if [[ $patch_size == 512 ]]; then
    curr_bag_size='1024'
  else
    curr_bag_size=$bag_size
  fi
  curr_batch_size=$batch_size
  grad_accum=0
fi

all_feat_dirs="/mnt/Xsky/lhl/wsi_ft_git/data_feat/TCGA_UNI2_features/pt_files"
all_feat_dirs="/mnt/Xsky/lhl/wsi_ft_git/data_feat/TCGA_all_UNI_224_Large/pt_files"
# all_feat_dirs='/mnt/Xsky/lhl/wsi_ft_git/data_feat/TCGA_titan_features/pt_files'
cmd="CUDA_VISIBLE_DEVICES=$gpuid python -m training.main_survival \\
--data_source ${all_feat_dirs} \\
--results_dir ${save_dir} \\
--split_dir ${split_dir} \\
--split_names ${split_names} \\
--task ${task} \\
--target_col ${target_col} \\
--model_type ${model} \\
--model_config ${model_config} \\
--n_fc_layers ${n_fc_layer} \\
--in_dim ${input_dim} \\
--opt ${opt} \\
--lr ${lr} \\
--lr_scheduler ${curr_lr_scheduler} \\
--wd ${wd} \\
--warmup_epochs ${warmup} \\
--max_epochs ${max_epoch} \\
--train_bag_size ${curr_bag_size} \\
--batch_size ${curr_batch_size} \\
--in_dropout 0.1 \\
--seed 1 \\
--num_workers 8 \\
--loss_fn ${loss_fn} \\
--nll_alpha ${alpha} \\
--n_label_bins ${n_label_bin} \\
--early_stopping ${es_flag} \\
--statistic ${statistic} \\
"

eval "$cmd"