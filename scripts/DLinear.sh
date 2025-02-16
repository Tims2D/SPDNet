#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
# Set parameters
label_len=48

model_name=DLinear
task_name=Multivariate_forecasting
root_path_name=./dataset/
data_path_name=aggregated_data.csv
data_name=aggregated_data
model_id_name=aggregated_data
random_seed=2024

seq_len_list=(720)
pred_len_list=(1 4 24 48 96)
down_sampling_layers=3
down_sampling_window=2

# Create necessary directories if they do not exist


for seq_len in "${seq_len_list[@]}"; do
    # Create necessary directories if they do not exist
    log_dir=logs/$model_name/$data_name/$seq_len

    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
    fi

    for pred_len in "${pred_len_list[@]}"; do
        echo "Running with pred_len: $pred_len"
        # Dynamically setting model_id_name to uniquely identify each run
        model_id_name="${model_name}_${data_name}_seq${seq_len}_pred${pred_len}"


        python -u arguments.py \
          --data_path $data_path_name \
          --data $data_name \
          --task_name $task_name \
          --random_seed $random_seed \
          --is_training 1 \
          --root_path $root_path_name \
          --model_id $model_id_name \
          --model $model_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 6 \
          --c_out 6 \
          --e_layers 2 \
          --n_heads 16 \
          --d_model 32 \
          --d_ff 32 \
          --dropout 0.25 \
          --fc_dropout 0.15 \
          --des 'Exp' \
          --train_epochs 25\
          --patience 5 \
          --lradj 'TST' \
          --target 'Load_demand' \
          --top_k 5 \
          --batch_size 32 \
          --learning_rate 0.0001 \
          --itr 1 >"$log_dir/${model_id_name}.log"

    done
done

