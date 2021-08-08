import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_words, embedding_size, lstm_cell_size, num_layers,
                 bidirectional=False, num_classes=2, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(num_words, embedding_size)
        self.bidir = bidirectional
        self.lstm = nn.LSTM(embedding_size, lstm_cell_size, num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(lstm_cell_size, num_classes)

    def forward(self, input):
        lengths, x = input[0], input[1:]
        x = self.embed(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        if self.bidir:
            out = self.linear(torch.cat([hidden[-2], hidden[-1]]))
        else:
            out = self.linear(hidden[-1])
        return out
