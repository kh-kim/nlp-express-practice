import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim,
            output_dim,
            n_layers,
            dropout,
            pad_idx,
        ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True, # If False, input shape is (seq_len, batch_size, input_size).
        )

        # Note that we use "hidden_dim * 2" since we are using bidirectional LSTM.
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, seq_len)

        embedded = self.dropout(self.embedding(x))
        # |embedded| = (batch_size, seq_len, embedding_dim)

        output, (hidden, cell) = self.lstm(embedded)
        # |output| = (batch_size, seq_len, hidden_dim * 2)
        # |hidden| = (n_layers * 2, batch_size, hidden_dim)
        # |cell| = (n_layers * 2, batch_size, hidden_dim)

        hidden = self.dropout(output[:, -1, :])
        # |hidden| = (batch_size, hidden_dim * 2)

        logit = self.fc(hidden)
        # |logit| = (batch_size, output_dim)

        log_prob = self.log_softmax(logit)
        # |log_prob| = (batch_size, output_dim)

        return log_prob
    
    def compute_loss(self, batch, criterion):
        x, y = batch["input_ids"], batch["labels"]
        # |x| = (batch_size, seq_len)
        # |y| = (batch_size, )

        device = next(self.parameters()).device

        x = x.to(device)
        y = y.to(device)

        logit = self.forward(x)
        # |logit| = (batch_size, output_dim)

        # softmax = nn.Softmax(dim=-1)
        # print(softmax(logit))

        loss = criterion(logit, y)
        # |loss| = (1, )

        return loss
