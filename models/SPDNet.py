__all__ = ['Times2D_3parts']

# Cell
from typing import Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from scipy.fft import rfft
import numpy as np
import math
from typing import Callable, Optional
from einops import rearrange
from layers.RevIN import RevIN
from layers.DecompositionTims2D import PastDecomposableMixing
from layers.Embed_Tims2D import DataEmbedding_wo_pos, DataEmbedding, DataEmbedding_inverted11

from layers.Transformer_EncDec_Times2D import Encoder, EncoderLayer, FullAttention, AttentionLayer



class Times2DBackbone(nn.Module):
    def __init__(self, configs, **kwargs):
        super(Times2DBackbone, self).__init__()
        self.configs = configs  # Make sure this line exists

        # Load parameters from configs
        self.channel_independence =1
        self.use_future_temporal_feature = 0
        self.layer = configs.e_layers
        self.data = configs.data
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.add = configs.add
        self.affine = configs.affine
        self.head_dropout = configs.head_dropout
        self.subtract_last = configs.subtract_last
        self.revin_layer = RevIN(self.enc_in, affine=self.affine, subtract_last=self.subtract_last)
        self.conv_blocks = nn.ModuleList()
        self.backbone = nn.ModuleList()
        self.n_layers = configs.e_layers
        self.wo_conv = configs.wo_conv
        self.serial_conv = configs.serial_conv
        self.n_heads = configs.n_heads
        self.d_ff = configs.d_ff
        self.attn_dropout = configs.attn_dropout
        self.kwargs = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flatten = nn.Flatten(start_dim=2)
        self.dropout = configs.dropout
        self.act = 'gelu'
        self.norm = 'BatchNorm'
        self.key_padding_mask = 'auto'
        self.padding_var = None
        self.attn_mask = None
        self.res_attention = True
        self.pre_norm = False
        self.store_attn = False
        self.pe = 'zeros'
        self.learn_pe = True
        self.verbose = False
        self.patch_len  = configs.patch_len
        self.batch = configs.batch_size
        self.task_name =configs.task_name
        self.fc_dropout = configs.fc_dropout
        # Define period_list and period_len
      
        #self.period_list = [720, 360, 140, 70, 48]  # weather 
        self.period_list = [720, 360, 110, 96, 48]  # M4 yearly EETT (h and m)
        

        #self.period_list = configs.period_list
        self.top_k = len(self.period_list)
        self.period_len = [math.ceil(self.seq_len / i) for i in self.period_list]
        # Define kernel_list and stride_list
        self.kernel_list = [(n, self.patch_len[i]) for i, n in enumerate(self.period_len)]
        self.stride_list = self.kernel_list

        # Define dim_list and tokens_list
        self.dim_list = [k[0] * k[1] for k in self.kernel_list]
        self.tokens_list = [
            (self.period_len[i] // s[0]) *
            ((math.ceil(self.period_list[i] / k[1]) * k[1] - k[1]) // s[1] + 1)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ]

        self.linear_layers = nn.ModuleList([
            nn.Linear(p_len * p_list, tokens * self.d_model) 
            for p_len, p_list, tokens in zip(self.period_len, self.period_list, self.tokens_list)
        ])
        
        self.conv2D = nn.ModuleList([
            nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s).to(self.device)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ])
        

        self.head = Head(self.seq_len, self.top_k, self.pred_len, head_dropout=self.head_dropout, Concat=not self.add).to(self.device)


        
        self.conv1D = nn.ModuleList([
            nn.Sequential(
                        *[
                        nn.Sequential(
                            # Depthwise convolution
                            nn.Conv1d(n, n, kernel_size=k, groups=n, padding=k // 2, bias=False),
                            nn.BatchNorm1d(n),
                            nn.SELU(),
                            # Pointwise convolution
                            nn.Conv1d(n, n, kernel_size=1, bias=False),
                            nn.BatchNorm1d(n),
                            nn.SELU()
                        ) for k in self.kernel_list if isinstance(k, int)
                    ],
                    nn.Dropout(self.fc_dropout),
                    nn.Flatten(start_dim=-2)
                ) for n in self.period_len
            ])

        
 

        
    ############################################### Itrasnformer  ##########################################################

        # Encoder-only architecture, aligned with ITransformer structure
        self.encoder = nn.ModuleList([
            Encoder(
                patch_num=token,  # Token from tokens_list
                d_model=self.d_model,  # Model dimensionality
                patch_len=self.dim_list[i],  # Patch length from dim_list
                pred_len=self.pred_len,  # Prediction length
                enc_in=self.enc_in,  # Encoder input size
                seq_len=self.seq_len,  # Sequence length
                attn_layers=[  # Define the attention layers
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention).to(self.device),
                            configs.d_model,
                            configs.n_heads
                        ).to(self.device),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ).to(self.device)
                    for l in range(configs.e_layers)  # Loop through the number of layers
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model).to(self.device)  # Layer normalization
            ).to(self.device)
            for i, token in enumerate(self.tokens_list)  # Loop through tokens_list and apply each token
        ])


    ############################################### TimeMixer ################################################################
         
        #self.dft_series_decomp = DFT_series_decomp(top_k=configs.top_k)
        #self.season_mixer = MultiScaleSeasonMixing(configs)
        #self.trend_mixer = MultiScaleTrendMixing(configs)
        self.fc_layer = nn.Linear(2 * self.pred_len, self.pred_len).to(self.device)

        
        self.predict_layers = torch.nn.Linear(
            configs.seq_len,  # input size
            configs.pred_len  # output size
        ).to(self.device)


        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True).to(self.device)
        
        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                     configs.dropout).to(self.device)

        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)]).to(self.device)

    def map_sequence_to_prediction(self, B, input_data, is_local=False):
        
            #enc_out_pdm shape from Itriansformer:
            #enc_out_pdm shape = torch.Size([B*N, Seq, N])
            
            #enc_out_pdm shape from Timmixer:
            #enc_out_pdm shape = torch.Size([[B*N, Seq, N])
            
        
        # for inputs [B*N, Seq, N]
        if not is_local:
            output = self.predict_layers(input_data.permute(0, 2, 1)).permute(0, 2, 1) # output = Size([B*N, Pred d_model])
            output = self.projection_layer(output)  # output  = torch.Size([B*N, Pred, 1])            
            if self.task_name == 'Multivariate_forecasting':
                output = output.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous() #zize([B, Pred, N])
            if self.task_name == 'Univariate_forecasting':
                output = output.reshape(B, 1, self.pred_len).permute(0, 2, 1).contiguous() #zize([B, Pred, N])
        # for the local input [B, N, Seq]
        else:
            
            output = self.predict_layers(input_data) # # output = Size([B, N, Pred])
            
 
        return output
################################################################################################################
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):    # x_enc [B, T, N]  x_mark_enc [B, T, N]
        

       
        #x_enc shape      = torch.Size([16, 96, 6])
        #x_mark_enc shape = torch.Size([16, 96, 4])
        #x_dec shape      = torch.Size([16, 144, 6])  ******* In timemixer X_dec = None
        #x_mark_dec shape = torch.Size([16, 144, 4])
        x =  x_enc
        #x = self.enc_embedding_timsnet(x_enc, x_mark_enc)     # [B,T,d or N], x shape = torch.Size([B, Seq, d_model])
        #x = self.linearEmbedding(x_enc, x_mark_enc)     # [B,T,d or N], x shape = torch.Size([B, Seq, d_model])
        
        
        B, T, d_emb = x.size()
        
           
        B, T, N = x_enc.size()       
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1) # x_enc shape= torch.Size([B * N, Seq, 1])
        x_mark_enc = x_mark_enc.repeat(N, 1, 1)
       
        '''
        x_enc shape= torch.Size([96, 96(seq), 1])
        x_mark_enc shape= torch.Size([96, 96(seq), 4])
        '''
        
        #Embedding 
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B*N,Seq,C]  enc_out shape= torch.Size([B * N, Seq, d_model])
        
        # Past Decomposable Mixing as encoder for past (with  embedded data)
        for i in range(self.layer):
            enc_out_pdm = self.pdm_blocks[i](enc_out) # enc_out_pdm shape = torch.Size([B*N, Seq, d_model])
            
        x_pdm = enc_out_pdm
            
            
        # Future Multipredictor Mixing as decoder for future
        dec_out = self.map_sequence_to_prediction(B, enc_out_pdm)  # dec_out shape = torch.Size([B, Pred_len, N])
        
        
        
        x = x.to(self.device)
        x = x.permute(0,2,1)
        B, N, T = x.size()
        
        res = []
        PDM_list = []
        for i, (period, (kernel_height, kernel_width)) in enumerate(zip(self.period_list, self.kernel_list)):
            # x input shape = torch.Size([16, 6, 96])
            
            '''
            period_list = [720, 360, 110, 96, 48]
            kernel_list = [(1, 48), (1, 32), (1, 16), (1, 6), (2, 3)]
            '''
            if self.seq_len % period != 0:
                pad1 = nn.ConstantPad1d((0, period - self.seq_len % period), 0)
                padded_X = pad1(x).to(self.device)
                padded_X = padded_X.reshape(padded_X.shape[0], padded_X.shape[1], padded_X.shape[2] // period, period)
            else:
                padded_X = x.reshape(B, N, T // period, period)
             
            if period % kernel_width != 0:
                pad2 = nn.ConstantPad1d((0, kernel_width - period % kernel_width), 0)
                out = pad2(padded_X).to(self.device)
            else:
                out = padded_X  # [B, N, patch, periods]
            
          
            '''
            out shape =  torch.Size([16, 6, 1, 720])
            out shape =  torch.Size([16, 6, 2, 384])
            out shape =  torch.Size([16, 6, 7, 112])
            out shape =  torch.Size([16, 6, 8, 96])
            out shape =  torch.Size([16, 6, 15, 48])
            '''
               
         
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])   # [B*N, Seq//period, period]
           
            '''
            out shape =  torch.Size([96, 1, 720])
            out shape =  torch.Size([96, 2, 384])
            out shape =  torch.Size([96, 7, 112])
            out shape =  torch.Size([96, 8, 96])
            out shape =  torch.Size([96, 15, 48])
            '''

            #____________________________________ Local and global __________________________________________
            Short_long = self.conv1D[i](out).reshape(x.shape[0], x.shape[1], -1)[..., :x.shape[-1]]
            
            
            '''
            Short_long shape in period = 720= torch.Size([16, 6, 720])
            Short_long shape in period = 360= torch.Size([16, 6, 720])
            Short_long shape in period = 110= torch.Size([16, 6, 720])
            Short_long shape in period = 96= torch.Size([16, 6, 720])
            Short_long shape in period = 48= torch.Size([16, 6, 720])
            ''' 
            local = self.map_sequence_to_prediction(B, Short_long,is_local=True)      # local = Size([B, N, Pred])
            
            
            long, attns = self.encoder[i](Short_long, attn_mask=None)            # [B, pred_length) N]
            long = long.permute(0,2,1)
            
            '''
            glo shape after Linear = torch.Size([16, 6, 96])
            glo shape after Linear = torch.Size([16, 6, 96])
            glo shape after Linear = torch.Size([16, 6, 96])
            glo shape after Linear = torch.Size([16, 6, 96])
            glo shape after Linear = torch.Size([16, 6, 96])
            '''
     
            #____________________________________ End Local and global __________________________________________
            
            ######################################### Periodical ##########################################################
            out = out.unsqueeze(-3)  
            
            '''
            out shape =  torch.Size([96, 1, 1, 720])
            out shape =  torch.Size([96, 1, 2, 384])
            out shape =  torch.Size([96, 1, 7, 112])
            out shape =  torch.Size([96, 1, 8, 96])
            out shape =  torch.Size([96, 1, 15, 48])
            '''
            
            out = self.conv2D[i](out)
           
            '''          
            out shape in period 720 = torch.Size([96, 48, 1, 15])
            out shape in period 360 = torch.Size([96, 64, 1, 12])
            out shape in period 110 = torch.Size([96, 112, 1, 7])
            out shape in period 96 = torch.Size([96, 48, 1, 16])
            out shape in period 48 = torch.Size([96, 45, 1, 16])
            '''
            
            out = self.flatten(out)
            
            '''
            out shape in period 720 = torch.Size([96, 48, 15])
            out shape in period 360 = torch.Size([96, 64, 12])
            out shape in period 110 = torch.Size([96, 112, 7])
            out shape in period 96 = torch.Size([96, 48, 16])
            out shape in period 48 = torch.Size([96, 45, 16])
            '''
            
            out = rearrange(out, '(b n) d p -> b n d p', b=x.size(0))  # Reshape back to [B, N, dim_list[i], tokens_list[i]]
            
            '''
            with d_model = 6
            out shape in period 720 = torch.Size([16, 6, 48, 15])
            out shape in period 360 = torch.Size([16, 6, 64, 12])
            out shape in period 110 = torch.Size([16, 6, 112, 7])
            out shape in period 96 = torch.Size([16, 6, 48, 16])
            out shape in period 48 = torch.Size([16, 6, 45, 16])
            
            
            with d_model = 16 (embedding)
            
            out shape= torch.Size([16, 16, 48, 15])
            out shape= torch.Size([16, 16, 32, 12])
            out shape= torch.Size([16, 16, 112, 7])
            out shape= torch.Size([16, 16, 48, 16])
            out shape= torch.Size([16, 16, 45, 16])
            '''
            ######################################## PDM #####################################
            PDM = out.permute(0,2,3,1).reshape(B,-1,N)
            
            '''
            PDM shape = torch.Size([16, 720, 6])
            PDM shape = torch.Size([16, 768, 6])
            PDM shape = torch.Size([16, 784, 6])
            PDM shape = torch.Size([16, 768, 6])
            PDM shape = torch.Size([16, 720, 6])
            
            '''
            PDM = PDM[:, :(self.seq_len), :].permute(0,2,1)     # [B, N, Seq]
            
            
            PDM = self.map_sequence_to_prediction(B, PDM,is_local=True) # local = Size([B, N, Pred])
            
            ######################################### End ##################################################################

            
            
            res.append(long + local + PDM)
            
        #PDM_list = torch.stack(PDM_list, dim=-1).sum(-1)  # [B, Seq, N]
        
        
        res = [r.to(self.device) for r in res]
        z = self.head(res)                                # [B, N, Pred_len] 
        dec_out = dec_out.permute(0,2,1)                  # [B, N, Pred_len]
       
        
        #combined  = z                                    # [B, N, Pred_len]

        
        combined = torch.cat((z, dec_out), dim=-1)        # [B, N, 2 * Pred_len]
        combined = self.fc_layer(combined)                # [B, N, Pred_len]                                 
        
        if self.task_name == 'Multivariate_forecasting':
            combined = combined[:, -1:, :]  # [B, 1, Pred_len]
        return combined
    
    
class Head(nn.Module):
    def __init__(self, seq_len, top_k, pred_len, head_dropout=0, Concat=True):
        super().__init__()
        self.Concat = Concat
        # Update the Linear layer with the new argument names
        self.linear = nn.Linear(pred_len * (top_k if Concat else 1), pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        '''
        x type= <class 'list'>
        x length= 5
        x[0] shape= torch.Size([16, 6, 96])
        '''
        
        if self.Concat:
            
            x = torch.cat(x, dim=-1)    # x shape  = torch.Size([16, 6, 480])  or [B, N, top_k*pred_len]
            
            x = self.linear(x)       # x shape after linear  = torch.Size([16, 6, 96])
           
        else:
            x = torch.stack(x, dim=-1)
            x = torch.mean(x, dim=-1)
            x = self.linear(x)
        x = self.dropout(x)
        return x
    
    
class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()
        self.model = Times2DBackbone(configs, **kwargs)

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):     #batch_x and batch_x_mark [B, T, N]
        x = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)     # [B, 1, Pred_len] 
        x = x.permute(0, 2, 1)                                           # [B, Pred_len, 1]  into the forecsating
        return x