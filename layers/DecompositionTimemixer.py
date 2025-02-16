import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class AdvancedSeriesDecomp(nn.Module):
    def __init__(self, trend_kernel_size, seasonal_kernel_size, input_channels):
        super(AdvancedSeriesDecomp, self).__init__()
        padding_trend = (trend_kernel_size - 1) // 2
        padding_seasonal = (seasonal_kernel_size - 1) // 2
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Trend extraction layer
        self.trend_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=trend_kernel_size,
            padding=padding_trend,
            groups=input_channels,  # Apply convolution across each input channel independently
            bias=False
        ).to(self.device)

        # Seasonal extraction layer
        self.seasonal_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=seasonal_kernel_size,
            padding=padding_seasonal,
            groups=input_channels,  # Apply convolution across each input channel independently
            bias=False
        ).to(self.device)

    def forward(self, x):
        """
        x: Input time series tensor of shape [Batch*N, Channels, Length]
        
        The inputs are like  :
        x shape =  torch.Size([96, 1, 720])
        x shape =  torch.Size([96, 2, 384])
        x shape =  torch.Size([96, 7, 112])
        x shape =  torch.Size([96, 8, 96])
        x shape =  torch.Size([96, 15, 48])
        """
       
        # Trend component
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

    
class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        self.d_ff = configs.d_ff
        self.trend_kernel_size = configs.trend_kernel_size
        self.seasonal_kernel_size = configs.seasonal_kernel_size
        

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

        

        HiddenSize= 64

    def forward(self, x_list):

        # only one sequence
        x = x_list
        B, N, T = x.size()
        
        '''
        if configs.decomp_method == 'moving_avg':
            self.decomposition = series_decomp(configs.moving_avg)
        '''
        # Initialize the decomposition dynamically for each input channel size N
        self.decomposition = AdvancedSeriesDecomp(
            trend_kernel_size=self.trend_kernel_size,
            seasonal_kernel_size=self.seasonal_kernel_size,
            input_channels=N  # Number of input channels
        )
        
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=N, out_features=self.d_ff),
            nn.GELU(),
            nn.Linear(in_features=self.d_ff, out_features=N),
        ).to(self.device)
        ##################################################### Decompose to obtain the season and trend
        
        #season, trend = self.decomposition(x)
        trend, season, residual = self.decomposition(x)   # [B, T, N]
                    # torch.Size([96, 96, 16])
                    # torch.Size([96, 96, 16])
        
        '''
        x      shape   = torch.Size([96, 1, 720])
        season shape   = torch.Size([96, 720, 1])
        trend shape    = torch.Size([96, 720, 1])
        residual shape = torch.Size([96, 720, 1])
        '''
        x = x.permute(0,2,1)
        out = season + trend + residual    
        out = self.out_cross_layer(out)
        out = x + out
        
        '''
        _____________________________ out_list _______________________________________
        out_list length          = 4
        out_list[0]  shape       = torch.Size([96, 96, 16])
        out_list[1]  shape       = torch.Size([96, 48, 16])
        out_list[2]  shape       = torch.Size([96, 24, 16])
        out_list[3]  shape       = torch.Size([96, 12, 16])
        '''
        out = out.permute(0,2,1)
        return out