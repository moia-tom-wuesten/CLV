import importlib
import pandas as pd
import os, sys, time, logging
import psutil

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)

import tensorflow_model.training_model as train_model
import helpers.pre_processing as preprocessing

importlib.reload(preprocessing)
importlib.reload(train_model)


def load_data(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "bank":
        print(dataset_name)
        df = pd.read_csv(
            filepath_or_buffer="trans.zip",
            usecols=["account_id", "date"],
            parse_dates=["date"],
        )
        df = df.rename(columns={"account_id": "customer_id"})
    elif dataset_name == "moia":
        df = pd.read_csv("moia_data.csv")
        df = df.drop(columns=["Unnamed: 0"])
        df["date"] = pd.to_datetime(df["date"])
    else:
        print("Wrong dataset_name")
    return df


def preprocess(
    df: pd.DataFrame,
    training_start,
    training_end,
    holdout_start,
    holdout_end,
    batch_train_size,
):
    prep = preprocessing.Preprocessing(
        df=df,
        training_start=training_start,
        training_end=training_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        batch_train_size=batch_train_size,
        name="benchmark",
    )
    train_dataloader, valid_dataloader = prep.run(dl_framwork="tensorflow")
    return train_dataloader, valid_dataloader, prep


def create_tf_dataloader(
    dataset_name,
    training_start,
    training_end,
    holdout_start,
    holdout_end,
    batch_train_size,
):
    df = load_data(dataset_name)

    train_dataloader, valid_dataloader, prep = preprocess(
        df, training_start, training_end, holdout_start, holdout_end, batch_train_size
    )
    return train_dataloader, valid_dataloader, prep


def load_tf_model(prep):
    model = train_model.TrainingModel(
        max_weeks=52,
        max_trans=12,
        seq_len=155,
        no_valid_samples=prep.no_valid_samples,
        no_train_samples=prep.no_train_samples,
        batch_size_train=prep.batch_train_size,
        batch_size_val=prep.no_valid_samples,
        max_epoch=10,
        name="benchmark",
    )
    return model


def train_tf(model, training_dataloader, validation_dataloader):
    list_of_cpu_usage = list()
    start_time = time.time()
    logging.info(f"Start Time: {start_time:.2f}")
    model.train_model(
        train_dataset=training_dataloader,
        valid_dataset=validation_dataloader,
    )
    list_of_cpu_usage.append(psutil.cpu_percent(4))
    elapsed = (time.time() - start_time) / 60
    print(elapsed)
    cpu_usage = sum(list_of_cpu_usage) / len(list_of_cpu_usage)
    logging.info(f"Total Time: {elapsed:.2f} min")
    return str(elapsed), str(cpu_usage)
