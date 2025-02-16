import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from layers.Embed_Tims2D import  DataEmbedding_inverted11
def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
        if pe == None:
            W_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
            nn.init.uniform_(W_pos, -0.02, 0.02)
            learn_pe = False
        elif pe == 'zero':
            W_pos = torch.empty((q_len, 1))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pe == 'zeros':
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pe == 'normal' or pe == 'gauss':
            W_pos = torch.zeros((q_len, 1))
            torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
        elif pe == 'uniform':
            W_pos = torch.zeros((q_len, 1))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)
        elif pe == 'lin1d':
            W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
        elif pe == 'exp1d':
            W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
        elif pe == 'lin2d':
            W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
        elif pe == 'exp2d':
            W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
        elif pe == 'sincos':
            W_pos = PositionalEncoding(q_len, d_model, normalize=True)
        else:
            raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
        return nn.Parameter(W_pos, requires_grad=learn_pe)
    
    
class iEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
    
class Encoder(nn.Module):
    def __init__(self, patch_num, d_model, patch_len, pred_len, enc_in, seq_len, 
                 attn_layers, conv_layers=None, norm_layer=None,
                  dropout=0., 
                  pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.patch_num = patch_num  # Token or patch_num for each encoder
        q_len = patch_num
        
        self.enc_in = enc_in
        self.patch_num = patch_num  # Token or patch_num for each encoder
        self.patch_len = patch_len  # Now patch_len is part of Encoder
        self.d_model = d_model  # d_model as a parameter
        self.pred_len = pred_len
        self.seq_len  = seq_len
        
        self.dropout = nn.Dropout(dropout)
        self.norm = norm_layer
        self.W_P = nn.Linear(self.patch_len, self.d_model)  
        self.W_pos = positional_encoding(pe, learn_pe, q_len, self.d_model)
        
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)
        
        self.predict_layers = torch.nn.ModuleList(
                [torch.nn.Linear(
                        self.seq_len ,
                        self.pred_len,)
                    for i in range(1)])
        
       
    
        
                                                    
    def future_multi_mixing(self, B, enc_out_pdm, x_enc):  
        
            dec_out = self.predict_layers[0](enc_out_pdm.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
            if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
            else:
                    dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()

            return dec_out
    
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        
       
        '''
        The input for Itrasnformer must be: torch.Size([96, 96, 16]) [B, T, N]
        The input frpm the periodical steps are like: 
        
        x shape in period 0 = torch.Size([16, 6, 48, 15])   [B, N, dim_list[i], tokens_list[i]]
        x shape in period 1 = torch.Size([16, 6, 32, 12])
        x shape in period 2 = torch.Size([16, 6, 16, 7])
        x shape in period 3 = torch.Size([16, 6, 6, 16])
        x shape in period 4 = torch.Size([16, 6, 6, 16])
        
        So they must be reshaped
        '''
        
        # Input encoding
        #x = x.permute(0, 1, 3, 2)  
        #x = self.W_P(x)  
        
        '''
        [B, N,  tokens_list[i], d_model]
        
        x shape= torch.Size([16, 6, 15, 16])
        x shape= torch.Size([16, 6, 12, 16])
        x shape= torch.Size([16, 6, 7,  16])
        x shape= torch.Size([16, 6, 16, 16])
        x shape= torch.Size([16, 6, 16, 16])
        '''
        #u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  
        
        '''
        u shape= torch.Size([96, 15, 16])   [B *N,  tokens_list[i], d_model]
        u shape= torch.Size([96, 12, 16])
        u shape= torch.Size([96, 7, 16])
        u shape= torch.Size([96, 16, 16])
        u shape= torch.Size([96, 16, 16])
        
        W_pos = self.W_pos.to(u.device)
        
        if u.shape[1] != W_pos.shape[0]:
            W_pos = positional_encoding('zeros', True, u.shape[1], W_pos.shape[1]).to(u.device)

        x = self.dropout(u + W_pos)  
       
        x shape =  torch.Size([96, 15, 16])
        x shape =  torch.Size([96, 12, 16])
        x shape =  torch.Size([96, 7, 16])
        x shape =  torch.Size([96, 16, 16])
        x shape =  torch.Size([96, 16, 16])
        '''
        #self.linearEmbedding = DataEmbedding_inverted11(u.size(1), self.d_model).to(u.device)       
        #x =  self.linearEmbedding(u)
        '''
        x shape = torch.Size([96, 16, 16])
        x shape = torch.Size([96, 16, 16])
        x shape = torch.Size([96, 16, 16])
        x shape = torch.Size([96, 16, 16])
        x shape = torch.Size([96, 16, 16])
        
        '''
        # x [B, T, N]  The input for Itrasnformer must be: torch.Size([96, 96, 16])
        ############################################ Times2D_3parts end ######################################
        x = x.permute(0,2,1)    #[B, Seq, N]
        
        self.linearEmbedding = DataEmbedding_inverted11(self.seq_len, self.d_model).to(x.device)
        x =  self.linearEmbedding(x)  #[B, N, , d_model]
        ############################################ Times2D_3parts end ######################################
        
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        
        # x has shape [B, N, , d_model] 
        
        #x = self.projector(x).permute(0, 2, 1)[:, :, :N] # filter the covariates
        '''
        x shape = torch.Size([96, 96, 6])
        x shape = torch.Size([96, 96, 6])
        x shape = torch.Size([96, 96, 6])
        x shape = torch.Size([96, 96, 6])
        x shape = torch.Size([96, 96, 6])
        '''
        '''
        With Itrnasformer embedding
        x shape = torch.Size([96, 96, 6])
        x shape = torch.Size([96, 96, 6])
        x shape = torch.Size([96, 96, 6])
        x shape = torch.Size([96, 96, 6])
        x shape = torch.Size([96, 96, 6])
        '''
        
        x = self.projector(x).permute(0,2,1)[:, :, :self.enc_in]     # [B, pred_len, , N] 
    
        return x, attns


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        '''
        self.norm2(x + y) shape = torch.Size([96, 15, 16])
        self.norm2(x + y) shape = torch.Size([96, 15, 16])
        self.norm2(x + y) shape = torch.Size([96, 12, 16])
        self.norm2(x + y) shape = torch.Size([96, 12, 16])
        self.norm2(x + y) shape = torch.Size([96, 7, 16])
        self.norm2(x + y) shape = torch.Size([96, 7, 16])
        self.norm2(x + y) shape = torch.Size([96, 16, 16])
        self.norm2(x + y) shape = torch.Size([96, 16, 16])
        self.norm2(x + y) shape = torch.Size([96, 16, 16])
        self.norm2(x + y) shape = torch.Size([96, 16, 16])

        '''
        return self.norm2(x + y), attn

    
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
        
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
