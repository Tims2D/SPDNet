#!/bin/bash

# Set CUDA device for GPU training
export CUDA_VISIBLE_DEVICES=0

# Set model and data parameters
model_name=Informer
task_name=Multivariate_forecasting
root_path_name=./dataset/
data_path_name=Load1.csv
model_id_name=Load1
data_name=Load1
random_seed=2024

label_len=48  # Adjust this as needed
seq_len_list=(96 384)
pred_len_list=(1 4 24 48 96)

for seq_len in "${seq_len_list[@]}"; do
    # Create necessary directories if they do not exist
    log_dir=logs/$model_name/$seq_len

    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
    fi

    for pred_len in "${pred_len_list[@]}"; do
        echo "Running with pred_len: $pred_len"
        # Dynamically setting model_id_name to uniquely identify each run
        model_id_name="${model_name}_${data_name}_seq${seq_len}_pred${pred_len}"
    python -u arguments.py \
        --random_seed $random_seed \
        --task_name $task_name \
        --model_id $model_id_name \
        --is_training 1 \
        --model $model_name \
        --data $data_name \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --dec_in 6 \
        --enc_in 6 \
        --c_out 6 \
        --e_layers 3 \
        --d_layers 1 \
        --d_model 16 \
        --dropout 0.4 \
        --train_epochs 50 \
        --patience 5 \
        --batch_size 32 \
        --target 'Load_demand'\
        --learning_rate 0.0001 \
        --itr 1 >"$log_dir/${model_id_name}.log"

    done
done
