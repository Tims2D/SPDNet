import torch
import torch.nn as nn

class ResLSTM_Block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(ResLSTM_Block, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.residual_connection = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        out = self.layer_norm(out)

        # Add residual connection
        residual = self.residual_connection(x)
        out += residual
        return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.task_name =configs.task_name
        # Define ResLSTM model layers
        self.res_lstm_block = ResLSTM_Block(
            input_size=configs.enc_in,    # Number of input features
            hidden_size=configs.d_model,  # Hidden state size
            num_layers=configs.e_layers,  # Number of LSTM layers
            dropout=configs.dropout       # Dropout rate
        )
        
        # Linear layer for projection to the output dimension
        self.projection = nn.Linear(configs.d_model, configs.c_out)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        x = x.to(self.device)
        # Pass through the ResLSTM block
        res_lstm_out = self.res_lstm_block(x)
        
        # Project the ResLSTM output to the desired output dimension
        out = self.projection(res_lstm_out[:, -self.pred_len:, :])  # out shape = [B, Pred_len, enc_in]
        out = out.permute(0,2,1)                                  # out shape = [B, enc_in, Pred_len]
        if self.task_name == 'Multivariate_forecasting':
            out = out[:, -1:, :]  # [B, 1, Pred_len]
        
        return out
