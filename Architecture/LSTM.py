import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout

        self.embelling_layer = nn.Linear(input_size, emb_size)

        self.forget_layer = nn.Linear(emb_size, hidden_size)
        self.input_layer  = nn.Linear(emb_size, hidden_size)
        self.cell_layer   = nn.Linear(emb_size, hidden_size)
        self.output_layer = nn.Linear(emb_size, hidden_size)

        self.hf = nn.Linear(hidden_size, hidden_size)
        self.hi = nn.Linear(hidden_size, hidden_size)
        self.hc = nn.Linear(hidden_size, hidden_size)
        self.ho = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.h2end = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_cell):
        hidden, cell = hidden_cell
        embedded = self.embelling_layer(input.float())

        forget = torch.sigmoid(self.forget_layer(embedded) + self.hf(hidden))
        input_gate = torch.sigmoid(self.input_layer(embedded) + self.hi(hidden))
        cell_state = torch.tanh(self.cell_layer(embedded) + self.hc(hidden))
        cell = forget * cell + input_gate * cell_state
        output = torch.sigmoid(self.output_layer(embedded) + self.ho(hidden))
        hidden = output * torch.tanh(cell)

        hidden = self.dropout(hidden)

        output = self.h2end(hidden)
        output = self.softmax(output)
        return output, hidden, cell

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
