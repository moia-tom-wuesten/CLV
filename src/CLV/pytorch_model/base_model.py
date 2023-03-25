import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        max_weeks,
        max_trans,
        stateful,
        hidden_size,
    ):
        super(LSTMModel, self).__init__()
        self.stateful = stateful
        self.hidden_size = hidden_size
        self.max_weeks = max_weeks
        self.max_trans = max_trans

        def emb_size(feature_max: int):
            return int(feature_max**0.5) + 1

        # Embeddings for LSTM
        self.embdedding_week = nn.Embedding(
            self.max_weeks, emb_size(self.max_weeks)
        )  # Shape(52,8)
        self.embedding_transaction = nn.Embedding(
            self.max_trans, emb_size(self.max_trans)  # Shape(12,4)
        )
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(
            input_size=emb_size(max_weeks) + emb_size(max_trans),
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_size, out_features=max_trans
        )
        self.softmax_layer = nn.Softmax(dim=2)

    def reset_hidden_state(self, x):
        self.hidden = (
            torch.zeros(1, x.size(0), self.hidden_size),
            torch.zeros(1, x.size(0), self.hidden_size),
        )

    def forward(self, x):
        if not self.stateful:
            self.reset_hidden_state(x)
        embedded_results = []
        x1 = torch.LongTensor(x[:, :, 0].unsqueeze(-1))
        x2 = torch.LongTensor(x[:, :, 1].unsqueeze(-1))
        embedded_results.append(self.embdedding_week(x1))
        embedded_results.append(self.embedding_transaction(x2))
        for i in range(2):
            embedded_results[i] = embedded_results[i].squeeze(2)
        stacked_tensors = torch.cat(embedded_results, 2)
        lstm_output, self.hidden = self.lstm(stacked_tensors, self.hidden)
        output = self.output_layer(lstm_output)
        if not self.stateful:
            return output  # Pytroch Lossfunction Cross Entropy has Softmax as Activation included -> no need to declare here
        else:
            return self.softmax_layer(output)
