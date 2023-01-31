import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class LSTMModel(pl.LightningModule):
    """
    A PyTorch Lightning module for training an LSTM model.

    Parameters:
        max_weeks (int): The maximum number of weeks in the dataset.
        max_trans (int): The maximum number of transactions in the dataset.
        stateful (bool): Whether the model should maintain hidden state between batches or not.
    """

    def __init__(self, max_weeks, max_trans, stateful):
        """
        Initialize the model with the given parameters.
        """
        super(LSTMModel, self).__init__()

        def emb_size(feature_max: int):
            return int(feature_max**0.5) + 1

        self.stateful = stateful
        self.max_weeks = max_weeks
        self.max_trans = max_trans
        self.embdedding_week = nn.Embedding(self.max_weeks, emb_size(self.max_weeks))
        self.embedding_transaction = nn.Embedding(
            self.max_trans, emb_size(self.max_trans)
        )
        self.lstm = nn.LSTM(
            input_size=12, hidden_size=128, dropout=0.5, batch_first=True
        )
        self.output_layer = nn.Linear(in_features=128, out_features=12)

    def reset_hidden_state(self, x):
        """
        Resets the hidden state of the LSTM to zeros.

        Parameters:
            x (torch.Tensor): Input tensor used to determine the number of samples in the batch.
        """
        self.hidden = (
            torch.zeros(1, x.size(0), 128).to(self.device),
            torch.zeros(1, x.size(0), 128).to(self.device),
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, num_classes).
        """
        if not self.stateful:
            self.reset_hidden_state(x)

        embedded_results = []
        x1 = x[:, :, 0].unsqueeze(-1)
        x2 = x[:, :, 1].unsqueeze(-1)

        embedded_results.append(self.embdedding_week(x1))
        embedded_results.append(self.embedding_transaction(x2))

        for i in range(2):
            embedded_results[i] = embedded_results[i].squeeze(2)

        stacked_tensors = torch.cat(embedded_results, 2)
        lstm_output, self.hidden = self.lstm(stacked_tensors, self.hidden)
        out = self.output_layer(lstm_output)
        return out

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        """
        Defines a single training step.

        Parameters:
            batch (tuple): A tuple of input and target tensors.
            batch_idx (int): The current batch index.

        Returns:
            dict: A dictionary containing the training loss.
        """
        x, y = batch
        y_hat = self(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_hat.permute(0, 2, 1), y.squeeze())
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Defines a single validation step.

        Parameters:
            batch (tuple): A tuple of input and target tensors.
            batch_idx (int): The current batch index.

        Returns:
            dict: A dictionary containing the validation loss.
        """
        x, y = batch
        y_hat = self(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_hat.permute(0, 2, 1), y.squeeze())

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes the average validation loss after an epoch.

        Parameters:
            outputs (list): A list of dictionaries, each containing the validation loss of a batch.

        Returns:
            dict: A dictionary containing the average validation loss.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        return {"val_loss": avg_loss}
