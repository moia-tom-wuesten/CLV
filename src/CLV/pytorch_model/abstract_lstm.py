import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from datetime import datetime

# import src.CLV.pytorch_model.base_model as LSTMModel
import pytorch_model.base_model as LSTMModel
import importlib
from torchviz import make_dot
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

importlib.reload(LSTMModel)


class EarlyStopping:
    """
    The EarlyStopping class provides a method to stop training early if there is no significant improvement in the validation loss.
    Attributes:
    tolerance (int): Number of consecutive epochs with no improvement to wait before stopping.
    min_delta (int): Minimum change in the monitored quantity to qualify as an improvement.
    counter (int): Number of consecutive epochs with no improvement.
    early_stop (bool): Flag indicating whether early stopping is triggered.
    """

    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        """
        Compares the difference between the train and validation loss and updates the counter and early_stop flag accordingly.
        """
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class Abstract_Lstm:
    def __init__(
        self,
        max_weeks: int,
        max_trans: int,
    ):
        super(Abstract_Lstm, self).__init__()
        self.max_weeks = max_weeks
        self.max_trans = max_trans
        self.train_hist = None
        self.val_hist = None
        self.best_model = None
        self.best_model_dict = None
        self.path = None

    def train(
        self,
        num_epochs,
        learning_rate,
        training_loader,
        validation_loader,
    ):
        early_stopping = EarlyStopping(tolerance=5, min_delta=0.5)
        model = LSTMModel.LSTMModel(
            max_weeks=self.max_weeks,
            max_trans=self.max_trans,
            stateful=False,
            hidden_size=128,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        best_mape_val = float("inf")
        self.best_model_dict = model.state_dict()
        best_epoch = 0
        self.train_hist = np.zeros(num_epochs)
        self.val_hist = np.zeros(num_epochs)
        softmax_layer = torch.nn.Softmax(dim=2)
        model.train()
        for epoch in range(num_epochs):
            if not epoch == 0:
                early_stopping(self.train_loss, self.val_loss)
                if early_stopping.early_stop:
                    print("We are at epoch:", epoch)
                    break
            for i, data in enumerate(training_loader):
                inputs, labels = data

                trainY_pred = model(inputs)  # predict train with the current model

                optimizer.zero_grad()
                self.train_loss = loss_function(
                    trainY_pred.permute(0, 2, 1), labels.squeeze()
                )
                # train_loss = loss_function(trainY_pred, torch.max(labels, dim=0)[0]) # compute the loss ("how bad is our model?")
                self.train_loss.backward()  # propagate the loss backwards through the network
                # print(train_loss.item())
                optimizer.step()  # update weights and biases
            self.train_hist[epoch] = self.train_loss.item()
            for i, vdata in enumerate(validation_loader):
                with torch.no_grad():
                    vinputs, vlabels = vdata
                    valY_pred = model(vinputs)
                    # val_loss=loss_function(valY_pred,torch.max(vlabels, dim=0)[0])
                    self.val_loss = loss_function(
                        valY_pred.permute(0, 2, 1), vlabels.squeeze()
                    )
                categorical = torch.distributions.Categorical(
                    probs=softmax_layer(valY_pred)
                )
                sample = categorical.sample().unsqueeze(-1)
                y_pred_val = np.array(sample.float())
                self.mape_val = mean_absolute_percentage_error(
                    np.array(vlabels.squeeze()), y_pred_val.squeeze()
                )
                # if self.val_loss < best_val_loss:
                #     best_val_loss = self.val_loss
                #     self.best_model_dict = model.state_dict()
                #     best_epoch = epoch
                if self.mape_val < best_mape_val:
                    best_mape_val = self.mape_val
                    self.best_model_dict = model.state_dict()
                    print(best_mape_val)
                    best_epoch = epoch
            self.val_hist[epoch] = self.val_loss.item()
            if epoch % 1000 == 999:
                print(
                    "Epoch: %d, loss: %1.5f, val_loss: %1.5f"
                    % (epoch, self.train_loss.item(), self.val_loss.item())
                )

        print("Best Epoch: %d, mape: %1.5f" % (best_epoch, best_mape_val))
        self.save_model(self.best_model_dict)
        self.load_best_model()

    def predict(self, batch):
        with torch.no_grad():
            self.best_model.eval()
            y_pred_dist = self.best_model(batch)
            categorical = torch.distributions.Categorical(probs=y_pred_dist)
            sample = categorical.sample().unsqueeze(-1)
        return np.array(sample.float())

    def get_mean_absolute_percentage_error(self, y_pred):
        y_pred = np.array(self.out_of_sample["transactions"])
        y = np.array(self.aggregate_counts["customer_id"][-self.holdout[0].shape[0] :])
        return mean_absolute_percentage_error(y, y_pred)

    def reset_cell_states(self, x):
        self.best_model.reset_hidden_state(x=x)

    def plot_losses(self):
        if self.train_hist is not None:
            plt.plot(self.train_hist, label="Training loss")
            plt.plot(self.val_hist, label="Validation loss")
            plt.legend()
        else:
            print("Please train the model first.")

    def get_model_summary(self):
        if self.best_model is not None:
            return self.best_model
        else:
            print("Please train the model first.")

    def save_model(self, model_dict):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        self.path = f"pytorch_model/saved_models/model_{dt_string}.pt"
        torch.save(
            {
                "model_state_dict": model_dict,
            },
            self.path,
        )

    def load_best_model(self):
        if self.path is not None:
            model_pred = LSTMModel.LSTMModel(
                max_weeks=self.max_weeks,
                max_trans=self.max_trans,
                stateful=True,
                hidden_size=128,
            )
            checkpoint = torch.load(self.path)
            model_pred.load_state_dict(checkpoint["model_state_dict"])
            self.best_model = copy.deepcopy(model_pred)
        else:
            print("No model to load.")

    def visualize_network(self, batch):
        if self.path is not None:
            y = self.best_model(batch)
            return make_dot(y.mean(), params=dict(self.best_model.named_parameters()))
        else:
            print("No model to load.")


if __name__ == "__main__":
    model = Abstract_Lstm(max_weeks=52, max_trans=12)
