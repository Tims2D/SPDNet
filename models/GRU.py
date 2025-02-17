import torch
import torch.nn as nn

class GRU_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(GRU_Block, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
       
    def forward(self, x):
        # GRU forward pass
        out, _ = self.gru(x)
        out = self.layer_norm(out)
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.task_name = configs.task_name
        # Define GRU model layers
        self.gru_block = GRU_Block(
            input_size=configs.enc_in,    # Number of input features
            hidden_size=configs.d_model,  # Hidden state size
            num_layers=configs.e_layers,  # Number of GRU layers
            dropout=configs.dropout       # Dropout rate
        )
        
        # Linear layer for projection to the output dimension
        self.projection = nn.Linear(configs.d_model, configs.enc_in)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        # Pass through the GRU block
        x = x.to(self.device)
        gru_out = self.gru_block(x)
        
        # Project the GRU output to the desired output dimension
        out = self.projection(gru_out[:, -self.pred_len:, :])  # out shape = [B, Pred_len, enc_in]
        out = out.permute(0, 2, 1)                             # out shape = [B, enc_in, Pred_len]
        if self.task_name == 'Multivariate_forecasting':
            out = out[:, -1:, :]  # [B, 1, Pred_len]
            out = out.permute(0,2,1)  # [B, Pred_len, 1]
        
        return out
