import argparse
import os
import sys
import torch
from exp.exp_forecasting import Exp_demand_Forecasting
import random
import numpy as np
from utils.str2bool import str2bool


parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer, LSTM, BiLSTM, RNN, ConvLSTM]')
parser.add_argument('--task_name', type=str, required=True, default='Multivariate_forecasting')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate '
                         'predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='Aggregated Load', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--scale', type=str, default=True, help='whether to use normalize; True 1 False 0')
parser.add_argument('--timeenc', type=int, default=1, help='Type of time encoding: 0 for manual encoding, 1 for learned encoding')
parser.add_argument('--flag', type=str, choices=['train', 'val', 'test'], help='Flag for data split')


parser.add_argument('--trend_kernel_size', type=int, default=25, help='Kernel size for the trend component')
parser.add_argument('--seasonal_kernel_size', type=int, default=7, help='Kernel size for the seasonal component')


# iTransformer
parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                    help='experiemnt name, options:[MTSF, partial_train]')
parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See 
parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                       'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

# SparseTSF
parser.add_argument('--period_len', type=int, default=24, help='period length')

#ModernTCN
parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='dmodels in each stage')
parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256])

parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')


# Mixer
parser.add_argument('--mix_time', type=int, default=1, help='mix_time')
parser.add_argument('--mix_variable', type=int, default=1, help='mix_variable')
parser.add_argument('--mix_channel', type=int, default=1, help='mix_channel')
parser.add_argument('--deform_patch', type=int, default=1, help='deform_patch')
parser.add_argument('--deform_range', type=float, default=0.25, help='deform_range')
parser.add_argument('--lambda_', type=float, default=1e-1, help='PaEn Weight')
parser.add_argument('--r', type=float, default=1e-2, help='Parameter of PaEn')
# Swin
parser.add_argument('--mlp_ratio', type=float, default=1.0, help='mlp_ratio')
parser.add_argument('--window_size', type=int, default=6, help='window_size')
parser.add_argument('--shift_size', type=int, default=3, help='shift_size')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='window_size')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.0, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--add', action='store_true', default=False, help='add')
parser.add_argument('--wo_conv', action='store_true', default=False, help='without convolution')
parser.add_argument('--serial_conv', action='store_true', default=False, help='serial convolution')

parser.add_argument('--kernel_list', type=int, nargs='+', default=[3, 7, 9], help='kernel size list')
parser.add_argument('--patch_len', type=int, nargs='+', default=[16], help='patch high')
parser.add_argument('--period', type=int, nargs='+', default=[24, 12], help='period list')
parser.add_argument('--stride', type=int, nargs='+', default=None, help='stride')

parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')

# ConvLSTM specific parameters
parser.add_argument('--kernel_size', type=int, nargs='+', default=(3, 3), help='kernel size for ConvLSTM')
parser.add_argument('--convlstm_kernel_size', type=int, default=3, help='kernel size for ConvLSTM')  # Kernel size for ConvLSTM
parser.add_argument('--conv2Dlstm_kernel_size', type=int, nargs=2, help='kernel size for ConvLSTM', default=(3, 3))
parser.add_argument('--num_layers', type=int, default=3, help='number of ConvLSTM layers')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden state size for ConvLSTM')

# Model configuration
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default 1: value patch_embedding + temporal patch_embedding + positional patch_embedding 2: value '
                         'patch_embedding + temporal patch_embedding 3: value patch_embedding + positional patch_embedding 4: value patch_embedding')

# FLinear
parser.add_argument('--train_mode', type=int,default=0)
parser.add_argument('--cut_freq', type=int,default=0)
parser.add_argument('--base_T', type=int,default=24)
parser.add_argument('--H_order', type=int,default=2)
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

parser.add_argument('--enc_in', type=int, default=7, help='global_encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of global_encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in global_encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn_dropout', type=float, default=0.05, help='attention dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# output log file
parser.add_argument('--log', type=str, default='./logs/LongForecasting/PatchTST_Electricity_336_96.log', help='path of output log file')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)
Exp = Exp_demand_Forecasting


if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
