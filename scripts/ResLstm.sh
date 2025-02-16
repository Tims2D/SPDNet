#!/bin/bash

# Set CUDA device for GPU training
export CUDA_VISIBLE_DEVICES=0
model_name=ResLSTM
task_name=Multivariate_forecasting
root_path_name=./dataset/
data_path_name=Load1.csv
model_id_name=Load1
data_name=Load1
random_seed=2024

seq_len=384
label_len=48  # Adjust this as needed
pred_len_list=(1 4 24 48 96)


## Create directories for logs if they don't exist
log_dir=logs/$model_name/$seq_len

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi


for pred_len in "${pred_len_list[@]}"; do    
    echo "Running with pred_len: $pred_len"
    model_id_name="${model_name}_${data_name}"
    
    python -u arguments.py \
        --random_seed $random_seed \
        --is_training 1 \
        --task_name $task_name \
        --model_id $model_id_name \
        --model $model_name \
        --data $data_name \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --enc_in 6 \
        --dec_in 6 \
        --c_out 6\
        --e_layers 3 \
        --d_model 8 \
        --dropout 0.4 \
        --train_epochs 50 \
        --patience 5 \
        --batch_size 32 \
        --target 'Load_demand' \
        --learning_rate 0.0001 \
        --itr 1 >"$log_dir/$model_id_name"_"$seq_len"_"$pred_len.log"
done