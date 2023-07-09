import importlib
import pandas as pd
import os, sys, time, logging
import psutil
import pyreadr
import json
import numpy as np
import tensorflow as tf

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)

import tensorflow_model.training_model as train_model
import tensorflow_model.prediction_model as pred_model
import helpers.pre_processing as preprocessing

importlib.reload(preprocessing)
importlib.reload(train_model)
importlib.reload(pred_model)


def load_data(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "bank":
        print(dataset_name)
        df = pd.read_csv(
            filepath_or_buffer="data/trans.zip",
            usecols=["account_id", "date"],
            parse_dates=["date"],
        )
        df = df.rename(columns={"account_id": "customer_id"})
    elif dataset_name == "moia":
        df = pd.read_csv("data/moia_data.csv")
        df = df.drop(columns=["Unnamed: 0"])
        df["date"] = pd.to_datetime(df["date"])
    elif dataset_name == "cdnow":
        cols = ["customer_id", "date", "amount", "revenue in $"]
        df = pd.read_fwf("data/CDNOW_master.txt", header=None, names=cols)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    elif dataset_name == "grocery":
        result = pyreadr.read_r("data/groceryElog.rda")
        df = pd.DataFrame(result["groceryElog"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"cust": "customer_id"})
        df["customer_id"] = df["customer_id"].astype("object")
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    elif dataset_name == "retail":
        df = pd.read_csv("data/uk_retail_cleaned.csv")
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


def load_tf_model(prep, dataset_name, train_dataloader, valid_dataloader):
    with open("feature_config.json", "r") as f:
        data = json.load(f)
    max_weeks = data[dataset_name]["features"]["weeks"]
    max_trans = data[dataset_name]["features"]["trans"]
    seq_len = data[dataset_name]["features"]["seq_len"]
    model = train_model.TrainingModel(
        max_weeks=max_weeks,
        max_trans=max_trans,
        seq_len=seq_len,
        no_valid_samples=prep.no_valid_samples,
        no_train_samples=prep.no_train_samples,
        batch_size_train=prep.batch_train_size,
        batch_size_val=prep.no_valid_samples,
        max_epoch=1,
        device="CPU",
        name="benchmark",
    )
    model.graph()
    model.train_model(train_dataloader, valid_dataloader)
    return model


def train_tf(model, training_dataloader, validation_dataloader):
    if tf.config.list_physical_devices("GPU"):
        print("GPU enabled")
    else:
        print("No GPU found")
    list_of_cpu_usage = list()
    list_of_performance_times = list()
    for i in range(10):
        start_time = time.time()
        model.train_model(
            train_dataset=training_dataloader,
            valid_dataset=validation_dataloader,
        )
        elapsed = time.time() - start_time
        list_of_cpu_usage.append(psutil.cpu_percent(4))
        list_of_performance_times.append(elapsed)
    mean_elapsed = np.mean(list_of_performance_times)
    min_elapsed = np.min(list_of_performance_times)
    std_elapsed = np.std(list_of_performance_times)
    print(f"{min_elapsed=}, {mean_elapsed=}, {std_elapsed=}")
    return (
        str(mean_elapsed),
        str(min_elapsed),
        str(std_elapsed),
        list_of_cpu_usage,
        list_of_performance_times,
    )


def inference(train_model, train_dataloader, valid_dataloader, prep):
    train_model.train_model(
        train_dataset=train_dataloader,
        valid_dataset=valid_dataloader,
    )
    prediction_model = pred_model.PredictionModel(
        prediction_batch_size=prep.no_valid_samples,
        model_weights_filename=train_model.model_weights_filename,
        training_model=train_model,
    )
    prediction_model.graph()
    list_of_cpu_usage = list()
    list_of_performance_times = list()
    for i in range(10):
        start_time = time.time()
        prediction_model.predict_model(valid_dataloader.take(1))
        elapsed = time.time() - start_time
        list_of_cpu_usage.append(psutil.cpu_percent(4))
        list_of_performance_times.append(elapsed)
    cpu_usage = sum(list_of_cpu_usage) / len(list_of_cpu_usage)
    mean_elapsed = np.mean(list_of_performance_times)
    min_elapsed = np.min(list_of_performance_times)
    std_elapsed = np.std(list_of_performance_times)
    print(f"{min_elapsed=}, {mean_elapsed=}, {std_elapsed=}")
    return (
        str(mean_elapsed),
        str(min_elapsed),
        str(std_elapsed),
        str(cpu_usage),
        list_of_performance_times,
        # list_of_cpu_usage,
    )


if __name__ == "__main__":
    import tensorflow as tf

    print(
        "Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU"))
    )
    # dataset_name = "grocery"
    # with open("feature_config.json", "r") as f:
    #     data = json.load(f)
    # training_start = data[dataset_name]["dates"]["training_start"]
    # training_end = data[dataset_name]["dates"]["training_end"]
    # holdout_start = data[dataset_name]["dates"]["holdout_start"]
    # holdout_end = data[dataset_name]["dates"]["holdout_end"]
    # train_dataloader, valid_dataloader, prep = create_tf_dataloader(
    #     dataset_name, training_start, training_end, holdout_start, holdout_end, 32
    # )
    # with open("feature_config.json", "r") as f:
    #     data = json.load(f)
    # max_weeks = data[dataset_name]["features"]["weeks"]
    # max_trans = data[dataset_name]["features"]["trans"]
    # seq_len = data[dataset_name]["features"]["seq_len"]
    # model = train_model.TrainingModel(
    #     max_weeks=max_weeks,
    #     max_trans=max_trans,
    #     seq_len=seq_len,
    #     no_valid_samples=prep.no_valid_samples,
    #     no_train_samples=prep.no_train_samples,
    #     batch_size_train=prep.batch_train_size,
    #     batch_size_val=prep.no_valid_samples,
    #     max_epoch=1,
    #     name="benchmark",
    #     device="CPU",
    # )
    # model.graph()
    # model.train_model(
    #     train_dataset=train_dataloader,
    #     valid_dataset=valid_dataloader,
    # )

    # prediction_model = pred_model.PredictionModel(
    #     prediction_batch_size=prep.no_valid_samples,
    #     model_weights_filename=model.model_weights_filename,
    #     training_model=model,
    # )
    # prediction_model.graph()
    # prediction_model.predict_model(valid_dataloader.take(1))
