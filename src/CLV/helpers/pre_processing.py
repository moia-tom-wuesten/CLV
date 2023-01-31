import pandas as pd
import numpy as np
import datetime
import random
from tqdm.auto import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader


class Preprocessing:
    def __init__(
        self,
        df: pd.DataFrame,
        training_start: datetime.datetime.date,
        training_end: datetime.datetime.date,
        holdout_start: datetime.datetime.date,
        holdout_end: datetime.datetime.date,
        batch_train_size: int,
    ):
        super(Preprocessing, self).__init__()
        self.df = df
        self.training_start = training_start
        self.training_end = training_end
        self.holdout_start = holdout_start
        self.holdout_end = holdout_end
        self.calender = pd.DataFrame({})
        self.holdout_calender = pd.DataFrame({})
        self.batch_train_size = batch_train_size
        self.batch_val_size = 0
        self.holdout = []
        self.no_train_samples = 0
        self.no_valid_samples = 0
        self.max_trans_cust = []
        self.calibration = []

    def date_range(self, start, end):
        """
        create a daterange
        """
        date_format = "%Y-%m-%d"
        start = datetime.datetime.strptime(start, date_format)
        end = datetime.datetime.strptime(end, date_format)
        r = (end + datetime.timedelta(days=1) - start).days
        return [start + datetime.timedelta(days=i) for i in range(r)]

    def create_calender(self) -> pd.DataFrame:
        """
        prepare a template for training samples to be filled with each customer's data
        """
        self.calender = pd.DataFrame(
            self.date_range(self.training_start, self.holdout_end), columns=["date"]
        )
        self.calender["year"] = self.calender["date"].dt.year  # 0-indexing
        self.calender["week"] = (self.calender["date"].dt.dayofyear // 7).clip(upper=51)

    def create_holdout_calender(self):
        """
        prepare a template for holdout samples to be filled with each customer's data
        """
        self.holdout_calendar = (
            self.calender[self.calender["date"] >= self.holdout_start]
            .drop(columns=["date"])
            .drop_duplicates()
            .drop(columns=["year"])
        )

    def check_dataframe(self):
        needed_columns = set(["date", "customer_id"])
        if needed_columns.issubset(set(self.df.columns)):
            print("TRUE")
        else:
            raise KeyError

    def split_trainingsdata(self, samples, targets):
        """
        split trainingsdata in trainigs and validatations data
        """
        VALIDATION_SPLIT = 0.1
        validation_size = round(len(samples) * VALIDATION_SPLIT)
        valid_samples, valid_targets = (
            samples[-validation_size:],
            targets[-validation_size:],
        )
        train_samples, train_targets = (
            samples[:-validation_size],
            targets[:-validation_size],
        )
        # number of samples in each dataset
        self.no_train_samples, self.no_valid_samples = len(train_samples), len(
            valid_samples
        )
        return train_samples, train_targets, valid_samples, valid_targets

    def transform_to_dataloaders(
        self, train_samples, train_targets, valid_samples, valid_targets
    ):
        """
        Transform into Data into pytorch Dataloaders
        """
        x_train = torch.Tensor(train_samples).long()
        y_train = torch.Tensor(train_targets).long()
        y_train = torch.unsqueeze(y_train, dim=-1)
        x_valid = torch.Tensor(valid_samples).long()
        y_valid = torch.Tensor(valid_targets).long()
        y_valid = torch.unsqueeze(y_valid, dim=-1)
        train_df = TensorDataset(x_train, y_train)
        train_dataloader = DataLoader(
            train_df, batch_size=self.batch_train_size, shuffle=True
        )
        valid_df = TensorDataset(x_valid, y_valid)
        valid_dataloader = DataLoader(
            valid_df, batch_size=self.batch_train_size, shuffle=True
        )
        return train_dataloader, valid_dataloader

    def create_customer_dataframe(self):
        trans = 0
        max_trans_per_week = 0

        samples = []
        targets = []
        # shuffle data randomly
        ids = self.df["customer_id"].unique()
        random.shuffle(ids)
        # build a record for each customer
        for account in tqdm(ids, desc="preparing dataset"):
            # take the data of single user,
            subset = (
                self.df.query("customer_id == @account")
                .groupby(["date"])
                .count()
                .reset_index()
            )
            user = subset.copy(deep=True)
            user = user.rename(columns={"customer_id": "transactions"})
            # copy the empty frame
            frame = self.calender.copy(deep=True)
            # insert customer ID
            frame["customer_id"] = account
            # merge customer data into the empty frame
            frame = frame.merge(user, on=["date"], how="left")
            # aggregate weekly transactions
            frame = (
                frame.groupby(["year", "week"])
                .agg({"transactions": "sum", "date": "min"})
                .sort_values(["date"])
                .reset_index()
            )
            # there is a small number of ids with 7 and more transactions per week
            # to make the job easier for the model, we can clip the value at 6:
            # frame['transactions'] = frame['transactions'].clip(upper=6)
            # this will however make the assertion following this block fail
            max_trans = max(frame["transactions"])
            max_trans_per_week = max(max_trans_per_week, max_trans)
            self.max_trans_cust.append(max_trans)
            # keep a count of the total transactions
            trans += user["transactions"].sum()
            # training sequences of everything until the holdout period
            training = frame[frame["date"] < self.holdout_start]
            training = training.drop(columns=["date", "year"]).astype(int)
            # store for later use
            self.calibration.append(training)
            # training sample: calibration sequence sans the final element
            sample = training[:-1].values  # not the last row
            # print(sample)
            samples.append(sample)
            # target labels: sequence of transaction counts starting
            # with the 2nd element. At each step of the training
            # we use a row from the train_samples element as our input,
            # and predict the corresponding element from train_targets
            target = training.loc[1:, "transactions"].values
            targets.append(target)
            # keep holdout sequence to compare with predictions
            hold = frame[frame["date"] >= self.holdout_start]
            hold = hold.drop(columns="date")
            self.holdout.append(hold)
        return samples, targets

    def run(self):
        self.check_dataframe()
        self.create_calender()
        samples, targets = self.create_customer_dataframe()
        (
            train_samples,
            train_targets,
            valid_samples,
            valid_targets,
        ) = self.split_trainingsdata(samples=samples, targets=targets)
        train_dataloader, valid_dataloader = self.transform_to_dataloaders(
            train_samples=train_samples,
            train_targets=train_targets,
            valid_samples=valid_samples,
            valid_targets=valid_targets,
        )
        self.create_holdout_calender()
        return train_dataloader, valid_dataloader
