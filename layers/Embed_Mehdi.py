import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)



class DataEmbedding_inverted_custom(nn.Module):
    def __init__(self, d_model, seq_len, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted_custom, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Tanh()

    def forward(self, x, x_mark):

        x = x.permute(0, 2, 1) # x after permute has shape : torch.Size([32, 7, 96])
        #print(f"x0 shape is {x.shape}")

        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:

            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)

            #print(f"x1 shape is {x.shape}")

            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(x) 
        # x: [Batch Variate d_model]
        return self.activation(x)




class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):

        x = x.permute(0, 2, 1) # x  has shape : torch.Size([32, 96, 7])
        
        if x_mark is None: # x_mark has shape : torch.Size([32, 96, 4])
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) # x after dataembedding_inverted has shape : torch.Size([32, 11, 256])

        return self.dropout(x)





class CustomLinearLayer18(nn.Module):
    def __init__(self, c_in, d_model):
        super(CustomLinearLayer18, self).__init__()
        self.linear = nn.Linear(c_in + 4, d_model)  # Adjusted to account for concatenation
        self.conv1d = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x, x_mark):

        print(f"Input x shape: {x.shape}")
        

        x1 = x[:, :48, :]
        x2 = x[:, 48:96, :]

        x_mark = x_mark[:, :48, :]

        x_linear = x1.permute(0, 2, 1)
        print(f"x_linear (after permute) shape: {x_linear.shape}")


        # Apply linear layer
        if x_mark is not None:
            x_mark_permuted = x_mark.permute(0, 2, 1)
            x_linear = torch.cat([x_linear, x_mark_permuted], dim=1)
        print(f"x_linear shape after concat [x_linear, x_mark_permuted] : {x_linear.shape}")
        
        batch_size, features, seq_len = x_linear.shape
        x_linear = x_linear.view(batch_size, features, -1).permute(0, 2, 1)
        x_linear = x_linear.reshape(-1, features)
        x_linear = self.linear(x_linear)
        x_linear = self.activation(x_linear)
        x_linear = x_linear.view(batch_size, seq_len, -1)
        print(f"x_linear (after linear and activation) shape: {x_linear.shape}")
        print(f"-------------------------------")

        # Apply conv1d layer
        print(f"x_conv (before conv1d) shape: {x2.permute(0, 2, 1).shape}")
        x_conv = self.conv1d(x2.permute(0, 2, 1))  # Apply conv1d to permuted input
        x_conv = self.activation(x_conv)
        print(f"x_conv (after activation) shape: {x_conv.shape}")
        
        x_conv = x_conv.permute(0, 2, 1)
        print(f"x_conv (after permute) shape: {x_conv.shape}")
        
        # Ensure the sequence length matches (pad or crop if necessary)
        seq_len_linear = x_linear.size(1)
        print(f"seq_len_linear shape is : {seq_len_linear}")
        seq_len_conv = x_conv.size(1)
        print(f"seq_len_conv shape is : {seq_len_conv}")
        if seq_len_linear != seq_len_conv:
            if seq_len_linear > seq_len_conv:
                pad_size = seq_len_linear - seq_len_conv
                x_conv = F.pad(x_conv, (0, 0, 0, pad_size))
                print(f"x_conv (after padding) shape: {x_conv.shape}")
            else:
                x_conv = x_conv[:, :seq_len_linear, :]
                print(f"x_conv (after cropping) shape: {x_conv.shape}")

        x = torch.cat((x_linear, x_conv), dim=1)
        
        print(f"x_final shape: {x.shape}")

        return x





### chatgpt and i Embedding

class CustomLinearLayer(nn.Module):
    def __init__(self, d_model, seq_len):
        super(CustomLinearLayer, self).__init__()
        self.linear = nn.Linear(seq_len, d_model)
        #self.activation = nn.ReLU()
        #self.activation = nn.GELU()
        #self.activation = nn.Sigmoid()
        self.activation = nn.Tanh()
        
    def forward(self, x, x_mark):

        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.linear(x)
        else:
            x = self.linear(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.activation(x)

    


class CustomLinearLayerconv1d2(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(CustomLinearLayerconv1d2, self).__init__()
        self.conv1d = nn.Conv1d(c_in + 4, d_model, kernel_size=3, padding=1)
        self.value_embedding = nn.Linear(c_in + 4 , d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        
        x = x.permute(0,2,1)

        if  x_mark is None:
            x = self.conv1d(x)
        else:

            x_mark_permuted = x_mark.permute(0, 2, 1)
            print(f" shape x_mark_permuted is: {x_mark_permuted.shape}")
            x_linear = torch.cat([x, x_mark_permuted], dim=1)
            print(f" shape x_linear after concat is: {x_linear.shape}")
            x_linear = self.conv1d(x_linear)
            print(f" shape x_linear after self.conv1d is: {x_linear.shape}")


        return self.dropout(x)


class DataEmbedding_inverted5(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted5, self).__init__()
        self.linear1 = nn.Linear(c_in + 89, d_model//2)
        self.linear2 = nn.Linear(d_model//2, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        #print(f" x shape is : {x.shape}")
        # x: [Batch Variate Time]
        if x_mark is None:

            x = self.linear1(x)
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)
        else:
        
            x = self.linear1((torch.cat([x, x_mark.permute(0, 2, 1)], 1)))
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)
      

        return self.dropout(x)


class DataEmbedding_inverted6(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted6, self).__init__()
        self.linear1 = nn.Linear(c_in + 89, d_model//4)
        self.linear2 = nn.Linear(d_model//4, d_model//2)
        self.linear3 = nn.Linear(d_model//2, d_model*3//2)
        self.linear4 = nn.Linear(d_model*3//2, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        #print(f" x shape is : {x.shape}")
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
        
            x = self.linear1((torch.cat([x, x_mark.permute(0, 2, 1)], 1)))
        
            x = torch.tanh(x)
        
            x = self.linear2(x)

            x = torch.tanh(x)

            x = self.linear3(x)

            x = torch.tanh(x)

            x = self.linear4(x)

            x = torch.tanh(x)
      

        return self.dropout(x)




class DataEmbedding_inverted7(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted7, self).__init__()
        self.linear1 = nn.Linear(c_in + 89, d_model*2)
        self.linear2 = nn.Linear(d_model*2, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):

        x = x.permute(0, 2, 1)

        #print(f" x shape is : {x.shape}")
 
        if x_mark is None:

            x = self.linear1(x)
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)


        else:
        
            x = self.linear1((torch.cat([x, x_mark.permute(0, 2, 1)], 1)))
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)
      

        return self.dropout(x)



class DataEmbedding_inverted7PEMS(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted7PEMS, self).__init__()
        self.linear1 = nn.Linear(c_in + 89, d_model*2)
        self.linear2 = nn.Linear(d_model*2, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):

        x = x.permute(0, 2, 1)

        #print(f" x shape is : {x.shape}")
 
        if x_mark is None:

            x = self.linear1(x)
            #x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)


        else:
        
            x = self.linear1((torch.cat([x, x_mark.permute(0, 2, 1)], 1)))
            #x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)
      

        return self.dropout(x)


class DataEmbedding_inverted8(nn.Module):
    def __init__(self, c_in, d_model, seq_len, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted8, self).__init__()
        self.linear1 = nn.Linear(seq_len, d_model*2)
        self.linear2 = nn.Linear(d_model*2, d_model*4)
        self.linear3 = nn.Linear(d_model*4, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        #print(f" x shape is : {x.shape}")
        # x: [Batch Variate Time]
        if x_mark is None:
            
            x = self.linear1(x)
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)

            x = self.linear3(x)
            x = torch.tanh(x)
            
        else:
        
            x = self.linear1((torch.cat([x, x_mark.permute(0, 2, 1)], 1)))
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)

            x = self.linear3(x)
            x = torch.tanh(x)
      

        return self.dropout(x)



## Na movafagh

class DataEmbedding_inverted9(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted9, self).__init__()
        self.conv1d = nn.Conv1d(in_channels= c_in + 89, out_channels= d_model*4, kernel_size= 3 , padding = 1)
        self.linear = nn.Linear(d_model*4, d_model)
        #self.batch_norm = nn.BatchNorm1d(d_model*2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        
        #print(f" x shape is : {x.shape}")

        #print(f" x_mark shape is : {x_mark.shape}")
       
        x_combind = torch.cat([x, x_mark], 2)

        if x_mark is None:
            x = self.value_embedding(x)
        else:
        
          # Apply Conv1d layer (input shape: [batch_size, input_dim, time_steps])
          x = self.conv1d(x_combind)
          #x = torch.tanh(x)
          #x = self.batch_norm(x)

          # Permute x to have shape [batch_size, time_steps, hidden_dim] for the linear layer
          x = x.permute(0, 2, 1)

          # Apply the linear layer
          x = self.linear(x)
          x = torch.tanh(x)
      

        return self.dropout(x)

#74

class DataEmbedding_inverted10(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted10, self).__init__()
        self.linear1 = nn.Linear(c_in + 89, d_model*2)
        self.linear2 = nn.Linear(d_model*2, d_model*4)
        self.linear3 = nn.Linear(d_model*4, d_model)
        self.linear4 = nn.Linear(c_in + 89, d_model*8)
        self.linear5 = nn.Linear(d_model*8, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        x_cat = torch.cat([x, x_mark.permute(0, 2, 1)], 1)
        #print(f" x shape is : {x.shape}")
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
        
            x = self.linear1(x_cat)
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)

            x = self.linear3(x)
            x = torch.tanh(x)

            x1 = self.linear4(x_cat)
            x1 = torch.tanh(x1)

            x1 = self.linear5(x1)
            x1 = torch.tanh(x1)

            x =(x1 + x)/2

        return self.dropout(x)



#373

class DataEmbedding_inverted11(nn.Module):
    def __init__(self, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted11, self).__init__()
        self.linear1 = nn.Linear(seq_len, d_model*2)
        self.linear2 = nn.Linear(d_model*2, d_model*4)
        self.linear3 = nn.Linear(d_model*4, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        
        x = x.permute(0, 2, 1)

        if x_mark is None:

            x = self.linear1(x)
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)

            x = self.linear3(x)
            x = torch.tanh(x)
      
        else:
        
            x = self.linear1((torch.cat([x, x_mark.permute(0, 2, 1)], 1)))
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)

            x = self.linear3(x)
            x = torch.tanh(x)
      
        return x


class DataEmbedding_inverted11_custom(nn.Module):
    def __init__(self, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted11_custom, self).__init__()
        self.linear1 = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        
        x = x.permute(0, 2, 1)

        if x_mark is None:

            x = self.linear1(x)
            #print(f"x shape is {x.shape}")
            x = torch.tanh(x)
            
      
        else:
        
            x = self.linear1((torch.cat([x, x_mark.permute(0, 2, 1)], 1)))
            #print(f"x1 shape is {x.shape}")
            x = torch.tanh(x)
        
      
        return x






class DataEmbedding_inverted11_11(nn.Module):
    def __init__(self, c_in, d_model, seq_len, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted11_11, self).__init__()

        self.linear1 = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, x_mark):
        
        x0 = x.permute(0, 2, 1)

        if x_mark is None:

            x1 = self.linear1(x0)
            x = torch.tanh(x1)

        else:

            x1 = torch.cat([x0, x_mark.permute(0, 2, 1)], 1)
            #print(f" x1 shape is : {x1.shape}") # x1 shape is : torch.Size([32, 11, 96])
            x2 = self.linear1(x1)
            x = torch.tanh(x2)  

        return self.dropout(x)





class DataEmbedding_inverted11_13(nn.Module):
    def __init__(self, c_in, d_model, seq_len, embed_type='fixed', freq='h'):
        super(DataEmbedding_inverted11_13, self).__init__()
        d_ff = d_model * 2
        self.linear1 = nn.Linear(seq_len, d_model)
        self.linear2 = nn.Linear(d_model*2, d_ff*2)
        self.linear3 = nn.Linear(d_ff*2, d_model)
        self.conv1 = nn.Conv1d(in_channels=seq_len,
                               out_channels=d_ff, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=3, padding=1)
        

    def forward(self, x, x_mark):
        
        x0 = x.permute(0, 2, 1)

        if x_mark is None:

            x1 = self.linear1(x0)
            x1 = torch.tanh(x1)
        
            x2 = self.linear2(x1)
            x2 = torch.tanh(x2)

            x3 = self.linear3(x2)
            x3 = torch.tanh(x3)

            x3 = x3.permute(0, 2, 1)

            x4 = self.conv1(x3)

            x = self.conv2(x4)
      
        else:
        
            x2 = torch.cat([x0, x_mark.permute(0, 2, 1)], 1)

            x3 = self.linear1(x2)
            x3 = torch.tanh(x3)
            # x1 shape is : torch.Size([32, 11, 96])
            #x4 = self.linear2(x3)
            #x4 = torch.tanh(x4)

            #x5 = self.linear3(x4)
            #x5 = torch.tanh(x5)

           
        return x3








class DataEmbedding_inverted11_14(nn.Module):
    def __init__(self, c_in, d_model, seq_len, embed_type='fixed', freq='h'):
        super(DataEmbedding_inverted11_14, self).__init__()
        self.linear1 = nn.Linear(seq_len, d_model)
        self.linear2 = nn.Linear(d_model//2, d_model)
        self.linear3 = nn.Linear(d_model*2, d_model)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(in_channels=d_model*2, out_channels=d_model*4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=d_model*4, out_channels=d_model*2, kernel_size=3, padding=1)

    def forward(self, x, x_mark):
        
        x = x.permute(0, 2, 1)

        if x_mark is None:

            x = self.linear1(x)
            x = torch.tanh(x)
        
            x = self.linear2(x)
            x = torch.tanh(x)

      
        else:
        
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)


            x1 = self.linear1(x)
            x1 = torch.tanh(x1)
        
            x2 = self.linear2(x1)
            x2 = torch.tanh(x2)


        return x



class DataEmbedding_inverted11_15(nn.Module):
    def __init__(self, c_in, d_model, seq_len, embed_type='fixed', freq='h'):
        super(DataEmbedding_inverted11_15, self).__init__()
        self.linear1 = nn.Linear(seq_len, d_model*2)
        self.linear2 = nn.Linear(d_model*2, d_model*4)
        self.linear3 = nn.Linear(d_model * 4, d_model)
        
        self.bn1 = nn.BatchNorm1d(d_model*2)
        self.bn2 = nn.BatchNorm1d(d_model*4)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(p=0.1)
        
        self.conv1 = nn.Conv1d(in_channels=d_model * 2, out_channels=d_model * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=d_model * 4, out_channels=d_model * 2, kernel_size=3, padding=1)

        print(f"d_model is {d_model}")

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, input_dim)

        if x_mark is None:
            x = self.linear1(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)
            x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)
            
            x = self.linear2(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)

            x = self.linear3(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)

        else:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)  # Concatenate along the feature dimension
            
            x = self.linear1(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)
            #print(f"x1 shape is : {x.shape}")
            x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)
            
            x = self.linear2(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)
            #print(f"x2 shape is : {x.shape}")
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)

            x = self.linear3(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)
            #print(f"x3 shape is : {x.shape}")
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)

        return x


class DataEmbedding_inverted11_16(nn.Module):
    def __init__(self, c_in, d_model, seq_len, embed_type='fixed', freq='h'):
        super(DataEmbedding_inverted11_16, self).__init__()
        self.linear1 = nn.Linear(seq_len, d_model)
        self.linear2 = nn.Linear(d_model*6, d_model)
        self.linear3 = nn.Linear(d_model * 4, d_model)
        
        self.bn1 = nn.BatchNorm1d(d_model*2)
        self.bn2 = nn.BatchNorm1d(d_model*4)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(p=0.1)
        
        self.conv1 = nn.Conv1d(in_channels=d_model * 2, out_channels=d_model * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=d_model * 4, out_channels=d_model * 2, kernel_size=3, padding=1)

        print(f"d_model is {d_model}")

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, input_dim)

        if x_mark is None:
            x = self.linear1(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)
            x = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)
            
            x = self.linear2(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)
            x = self.bn2(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)

            x = self.linear3(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)
            x = self.bn3(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = self.dropout(x)

        else:
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)  # Concatenate along the feature dimension
            
            x = self.linear1(x)  # (batch_size, d_model, input_dim)
            x = torch.sin(x)

            #x = self.linear2(x)  # (batch_size, d_model, input_dim)
            #x = torch.sin(x)


        return x



class Time2Vec2(nn.Module):
    def __init__(self, kernel_size):
        super(Time2Vec2, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, kernel_size - 1)
    
    def forward(self, x, x_mark):
        x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)
        v_linear = self.linear(x)
        v_periodic = torch.sin(self.periodic(x))
        return torch.cat([v_linear, v_periodic], dim=-1)




class Time2Vec(nn.Module):
    def __init__(self, c_in, kernel_size):
        super(Time2Vec, self).__init__()
        num_time_features = 4
        input_features = c_in + num_time_features
        self.linear = nn.Linear(input_features, 1)
        self.periodic = nn.Linear(input_features, kernel_size - 1)
    
    def forward(self, x, x_mark):
        # x: (batch_size, C, seq_len_x)
        # x_mark: (batch_size, seq_len_x_mark, num_time_features)
        
        # Ensure x_mark has the same sequence length as x
        seq_len_x = x.size(2)
        x_mark = x_mark[:, :seq_len_x, :]  # Adjust x_mark to have seq_len_x
        
        # Check shapes
        assert x.size(2) == x_mark.size(1), f"Sequence lengths do not match after adjustment: {x.size(2)} vs {x_mark.size(1)}"
        
        # Permute x to match x_mark's shape
        x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_len_x, C)
        
        # Concatenate along the feature dimension
        x = torch.cat([x, x_mark], dim=2)  # Shape: (batch_size, seq_len_x, C + num_time_features)
        
        # Flatten the feature dimensions
        batch_size, seq_len, channels = x.size()
        x_flat = x.view(-1, channels)  # Shape: (batch_size * seq_len_x, channels)
        
        # Apply linear transformations
        v_linear = self.linear(x_flat)  # Shape: (batch_size * seq_len_x, 1)
        v_periodic = torch.sin(self.periodic(x_flat))  # Shape: (batch_size * seq_len_x, kernel_size - 1)
        
        # Concatenate and reshape back
        v = torch.cat([v_linear, v_periodic], dim=1)  # Shape: (batch_size * seq_len_x, kernel_size)
        v = v.view(batch_size, seq_len_x, -1)  # Shape: (batch_size, seq_len_x, kernel_size)
        
        return v








    

