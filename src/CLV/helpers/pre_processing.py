import pandas as pd
import datetime
import random
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

import tensorflow as tf


class Preprocessing:
    def __init__(
        self,
        df: pd.DataFrame,
        training_start: datetime.datetime.date,
        training_end: datetime.datetime.date,
        holdout_start: datetime.datetime.date,
        holdout_end: datetime.datetime.date,
        batch_train_size: int,
        name: str,
    ):
        super(Preprocessing, self).__init__()
        self.df = df
        self.name = name
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
        self.seq_len = 0

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
            raise KeyError("Wrong pandas schema")

    def split_trainingsdata(self, samples, targets):
        """
        split trainingsdata in trainigs and validatations data
        """
        VALIDATION_SPLIT = 0.1
        self.seq_len = samples[0].shape[0]
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
        torch.manual_seed(42)

        # def seed_worker(worker_id):
        #     worker_seed = torch.initial_seed() % 2**32
        #     np.random.seed(worker_seed)
        #     random.seed(worker_seed)

        # g = torch.Generator()
        # g.manual_seed(42)
        train_dataloader = DataLoader(
            train_df,
            batch_size=self.batch_train_size,
            shuffle=True,
            drop_last=True,
            # worker_init_fn=seed_worker,
            num_workers=0,
            pin_memory=True,
            # generator=g,
        )

        valid_df = TensorDataset(x_valid, y_valid)
        if not self.name == "benchmark":
            torch.save(x_train, f"datasets/{self.name}_x_train.pt")
            torch.save(x_valid, f"datasets/{self.name}_x_valid.pt")
            torch.save(y_train, f"datasets/{self.name}_y_train.pt")
            torch.save(y_valid, f"datasets/{self.name}_y_valid.pt")
        valid_dataloader = DataLoader(
            valid_df,
            batch_size=self.batch_train_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
            # worker_init_fn=seed_worker,
            # generator=g,
        )

        print(f"Shape of x_train: {x_train.shape} Shape of x_val: {x_valid.shape}")
        return train_dataloader, valid_dataloader

    def decode_sample(self, sample, target):
        # expand dims is important to add a dimension for the batches
        return (
            {
                "week": tf.cast(tf.expand_dims(sample[:, 0], axis=-1), "int32"),
                "transaction": tf.cast(tf.expand_dims(sample[:, 1], axis=-1), "int32"),
            },
            tf.cast(tf.expand_dims(target, axis=-1), "int32"),
        )

    def transform_to_tf_datasets(
        self, train_samples, valid_samples, train_targets, valid_targets
    ):

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((train_samples, train_targets))
            .map(self.decode_sample)
            .batch(self.batch_train_size)
            .repeat()
        )

        valid_dataset = (
            tf.data.Dataset.from_tensor_slices((valid_samples, valid_targets))
            .map(self.decode_sample)
            .batch(self.no_valid_samples)
            .repeat()
        )

        return train_dataset, valid_dataset

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
            hold = hold.drop(columns=["date", "year", "week"])
            self.holdout.append(hold)
        return samples, targets

    def run(self, dl_framwork: str):
        self.check_dataframe()
        self.create_calender()
        self.create_holdout_calender()
        samples, targets = self.create_customer_dataframe()
        (
            train_samples,
            train_targets,
            valid_samples,
            valid_targets,
        ) = self.split_trainingsdata(samples=samples, targets=targets)
        if dl_framwork == "pytorch":
            train_dataloader, valid_dataloader = self.transform_to_dataloaders(
                train_samples=train_samples,
                train_targets=train_targets,
                valid_samples=valid_samples,
                valid_targets=valid_targets,
            )
            return train_dataloader, valid_dataloader
        elif dl_framwork == "tensorflow":
            train_dataset, valid_dataset = self.transform_to_tf_datasets(
                train_samples=train_samples,
                valid_samples=valid_samples,
                train_targets=train_targets,
                valid_targets=valid_targets,
            )
            return train_dataset, valid_dataset
        else:
            "Please choose between 'tensorflow' or 'pytorch'."
