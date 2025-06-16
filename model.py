# model.py

import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, output_size=1, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))  # out: (batch_size, seq_len, hidden_size)
        return self.fc(out[:, -1, :])      # Take last time step
