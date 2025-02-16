import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class AdvancedSeriesDecomp(nn.Module):


    def __init__(self, trend_kernel_size, seasonal_kernel_size, input_channels=1):
        super(AdvancedSeriesDecomp, self).__init__()
        padding_trend = (trend_kernel_size - 1) // 2
        padding_seasonal = (seasonal_kernel_size - 1) // 2

        # Trend extraction layer
        self.trend_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=trend_kernel_size,
            padding=padding_trend,
            groups=input_channels,
            bias=False
        )

        # Seasonal extraction layer
        self.seasonal_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=seasonal_kernel_size,
            padding=padding_seasonal,
            groups=input_channels,
            bias=False
        )

        # Initialize convolutional kernels
        nn.init.constant_(self.trend_conv.weight, 1.0 / trend_kernel_size)
        nn.init.constant_(self.seasonal_conv.weight, -1.0 / seasonal_kernel_size)

    def forward(self, x):
        """
        x: Input time series tensor of shape [Batch, Channels, Length]
        """
        # Trend component
        x = x.permute(0, 2, 1)  # Now x has shape [Batch, Channels, Length]
        
        trend = self.trend_conv(x)
        # Seasonal component
        seasonal = self.seasonal_conv(x - trend)

        # Residual component
        residual = x - trend - seasonal
        '''
        trend shape    = torch.Size([96, 16, 720])
        seasonal shape   = torch.Size([96, 16, 720])
        residual shape = torch.Size([96, 16, 720])
        '''
        trend    = trend.permute(0,2,1)
        seasonal = seasonal.permute(0,2,1)
        residual = residual.permute(0,2,1)
                
        return trend, seasonal, residual



class DFT_series_decomp(nn.Module):
    """
    Series decomposition block that applies DFT to extract seasonal and trend components.
    """
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        # Perform real FFT on the input time series
        xf = torch.fft.rfft(x, dim=-1)
        freq = abs(xf)
        freq[..., 0] = 0  # Remove the 0th frequency component (DC component)

        # Keep top_k frequencies and zero out the others
        top_k_freq, _ = torch.topk(freq, self.top_k, dim=-1)
        xf[freq <= top_k_freq.min()] = 0

        # Inverse FFT to obtain the seasonal component
        x_season = torch.fft.irfft(xf, n=x.size(-1), dim=-1)
        x_trend = x - x_season  # Trend is the residual after removing the seasonality
        return x_season, x_trend

    
class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]
            
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list
############################################################################################################################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# Define TemporalBlock
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# Define TemporalConvNet
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


    
class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        '''
        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        '''
        self.decompsition = AdvancedSeriesDecomp(
                            trend_kernel_size=25,
                            seasonal_kernel_size=7,
                            input_channels=configs.d_model  # Set to the number of features N
                                )
        

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

        HiddenSize= 64

    def forward(self, x_list):
        
        '''
        ____________________________ x_list _______________________________________
        x_list length          = 4
        x_list[0]  shape       = torch.Size([96, 96, 16])
        x_list[1]  shape       = torch.Size([96, 48, 16])
        x_list[2]  shape       = torch.Size([96, 24, 16])
        x_list[3]  shape       = torch.Size([96, 12, 16])
        '''
        '''
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
        '''
        # only one sequence
        x = x_list
        _, T, _ = x.size()
        '''
        length_list[0]  len       = 96
        length_list[1]  len       = 48
        length_list[2]  len       = 24
        length_list[3]  len       = 12
        '''
        
        ##################################################### Decompose to obtain the season and trend
        '''
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        '''
        # only one sequence 
        #season, trend = self.decompsition(x)
        trend, season, residual = self.decompsition(x)   # [B, T, N]
                    # torch.Size([96, 96, 16])
                    # torch.Size([96, 96, 16])
        
        '''        _____________________________ season _______________________________________
        season_list length          = 4
        season_list[0]  shape       = torch.Size([96, 16, 96])
        season_list[1]  shape       = torch.Size([96, 16, 48])
        season_list[2]  shape       = torch.Size([96, 16, 24])
        season_list[3]  shape       = torch.Size([96, 16, 12])
        _____________________________ trend _______________________________________
        trend_list length          = 4
        trend_list[0]  shape       = torch.Size([96, 16, 96])
        trend_list[1]  shape       = torch.Size([96, 16, 48])
        trend_list[2]  shape       = torch.Size([96, 16, 24])
        trend_list[3]  shape       = torch.Size([96, 16, 12])
        '''
        
        ########################################## bottom-up season mixing
        #out_season_list = self.mixing_multi_scale_season(season_list)
        
        '''
        out_season_list _______________________________________
        out_season_list length          = 4
        out_season_list[0]  shape       = torch.Size([96, 96, 16])
        out_season_list[1]  shape       = torch.Size([96, 48, 16])
        out_season_list[2]  shape       = torch.Size([96, 24, 16])
        out_season_list[3]  shape       = torch.Size([96, 12, 16])
        '''

        # top-down trend mixing
        #out_trend_list = self.mixing_multi_scale_trend(trend_list)
        '''
        _____________________________ out_trend_list _______________________________________
        out_trend_list length          = 4
        out_trend_list[0]  shape       = torch.Size([96, 96, 16])
        out_trend_list[1]  shape       = torch.Size([96, 48, 16])
        out_trend_list[2]  shape       = torch.Size([96, 24, 16])
        out_trend_list[3]  shape       = torch.Size([96, 12, 16])
        '''
        
        '''
        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        '''   
        #residual_output = self.residual_model(residual)  # [Batch, T, N]
        out = season + trend + residual
        out = x + self.out_cross_layer(out)
        
        '''
        _____________________________ out_list _______________________________________
        out_list length          = 4
        out_list[0]  shape       = torch.Size([96, 96, 16])
        out_list[1]  shape       = torch.Size([96, 48, 16])
        out_list[2]  shape       = torch.Size([96, 24, 16])
        out_list[3]  shape       = torch.Size([96, 12, 16])
        '''
        
        return out