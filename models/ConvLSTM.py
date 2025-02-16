import torch
import torch.nn as nn

class ConvLSTM_Block(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, dropout=0.5):
        super(ConvLSTM_Block, self).__init__()
        
        # 1D Convolution Layer
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Input x shape: [Batch, Time, Features]
        
        # Transpose x to [Batch, Features, Time] for Conv1D
        x = x.transpose(1, 2)
        
        # Apply 1D Convolution
        x = self.conv1d(x)
        
        # Transpose x back to [Batch, Time, Hidden_Dim]
        x = x.transpose(1, 2)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Layer Normalization
        x = self.layer_norm(x)
        
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.task_name =configs.task_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define ConvLSTM Block
        self.conv_lstm_block = ConvLSTM_Block(
            input_dim=configs.enc_in,         # Number of input features
            hidden_dim=configs.d_model,       # Hidden state size
            kernel_size=configs.convlstm_kernel_size,  # Convolution kernel size
            num_layers=configs.e_layers,      # Number of LSTM layers
            dropout=configs.dropout           # Dropout rate
        ).to(self.device)
        
        # Linear layer for projection to the output dimension
        self.projection = nn.Linear(configs.d_model, configs.enc_in).to(self.device)

    def forward(self, x):
        # Pass through the ConvLSTM block
        conv_lstm_out = self.conv_lstm_block(x)
        
        # Project the ConvLSTM output to the desired output dimension
        out = self.projection(conv_lstm_out[:, -self.pred_len:, :])  # [B, Pred_len, N]
        if self.task_name == 'Multivariate_forecasting':
            out = out[:, :, -1:]  # [B, ,Pred_len, 1]
            
        return out
