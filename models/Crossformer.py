import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from models.PatchTST import FlattenHead


from math import ceil


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0).to(self.device)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model)).to(self.device)
        self.pre_norm = nn.LayerNorm(configs.d_model).to(self.device)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l is 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
                            ) for l in range(configs.e_layers)
            ]
        ).to(self.device)
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model)).to(self.device)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model, configs.n_heads,
                                           configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    # activation=configs.activation,
                )
                for l in range(configs.e_layers + 1)
            ],
        ).to(self.device)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
      
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        out = dec_out[:, -self.pred_len:, :]  # [B, L=Pred_len, D]
       
        
        if self.task_name == 'Multivariate_forecasting':
            out = out[:, :, -1:]  # [B, ,Pred_len, 1]

        return out 
       