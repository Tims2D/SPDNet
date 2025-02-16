import torch.nn as nn
import torch
class Head(nn.Module):
    def __init__(self, context_window, num_period, target_window, head_dropout=0,
                 Concat=True):
        super().__init__()
        self.Concat = Concat
        self.linear = nn.Linear(context_window * (num_period if Concat else 1), target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.Concat:
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
        else:
            x = torch.stack(x, dim=-1)
            x = torch.mean(x, dim=-1)
            x = self.linear(x)
        x = self.dropout(x)
        return x