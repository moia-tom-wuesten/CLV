import argparse
import time
import importlib
import os
import sys
import warnings
import logging
import json
from typing import Optional, Tuple
import pandas as pd
from datetime import datetime
import torch


from train_pytorch_model import (
    load_pytorch_model,
    get_dataloaders_bank,
    train,
    get_pytorch_version,
    inference,
)

# from tf_utils import create_tf_dataloader, load_tf_model, train_tf, inference


def single_test(
    file_path: str,
    framework: str,
    batch_size: int,
    mode: str,
    device: str,
    dataset_name: str,
) -> None:
    """
    Run single time and GPU memory measuring with provided parameters.
    :param file_path: path to log csv file with measured time and memory.
    :param framework: "tf1", "tf2" ,"pytorch1" or "pytorch2".
    :param input_size: input image size.
    :param batch_size: batch size.
    :param mode: "train", "inference_gpu" or "inference_cpu".
    :param gpu_number: GPU index or None for using CPU.
    :param random_seed: random generators seed.
    :param steps_number: number of steps for testing.
    """
    # Disable TensorFlow logging.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    with open("feature_config.json", "r") as f:
        data = json.load(f)
    training_start = data[dataset_name]["dates"]["training_start"]
    training_end = data[dataset_name]["dates"]["training_end"]
    holdout_start = data[dataset_name]["dates"]["holdout_start"]
    holdout_end = data[dataset_name]["dates"]["holdout_end"]
    print("in single step")
    if framework == "tf2":
        print("in TF")

        # from tf_torch_models.tf_2x_model import TensorFlow2Model
        from tf_utils import create_tf_dataloader, load_tf_model, train_tf, inference

        train_dataloader, valid_dataloader, prep = create_tf_dataloader(
            dataset_name=dataset_name,
            training_start=training_start,
            training_end=training_end,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            batch_train_size=batch_size,
        )
        model = load_tf_model(
            prep=prep,
            dataset_name=dataset_name,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
        )
        input_shape = (128, 155, 2)
        # odel_object = TensorFlow2Model(random_seed, steps_number, gpu_number)
    elif framework == "pytorch1":
        # Disable UserWarning in PyTorch 1.9.0.
        print("Load model from PyTorch 1.9.0")
        warnings.filterwarnings("ignore", category=UserWarning)
        model = load_pytorch_model(dataset_name=dataset_name, device=device)
        train_dataloader, valid_dataloader = get_dataloaders_bank(
            dataset_name=dataset_name, batch_size=batch_size, num_workers=0
        )
        feature, target = next(iter(train_dataloader))
        input_shape = (feature.shape[0], feature.shape[1], feature.shape[2])
        # model_object = PyTorchModel(random_seed, steps_number, gpu_number)
    elif framework == "pytorch2":
        model = load_pytorch_model(dataset_name=dataset_name, device=device)
        model = model.to(device)
        train_dataloader, valid_dataloader = get_dataloaders_bank(
            dataset_name=dataset_name, batch_size=batch_size, num_workers=0
        )
        feature, target = next(iter(train_dataloader))
        input_shape = (feature.shape[0], feature.shape[1], feature.shape[2])
    elif framework == "pytorch2_compiled":
        model = load_pytorch_model(dataset_name=dataset_name)
        model = torch.compile(model)
        train_dataloader, valid_dataloader = get_dataloaders_bank(
            dataset_name=dataset_name, batch_size=batch_size, num_workers=0
        )
        feature, target = next(iter(train_dataloader))
        input_shape = (feature.shape[0], feature.shape[1], feature.shape[2])
    else:
        raise ValueError(
            f'Wrong framework "{framework}". Must be "tf1", "tf2" or "pytorch".'
        )

    memory, res_time, exc_info = "-", "-", ""

    if mode == "train":
        if framework == "tf2":
            mean_time, min_time, std_time, memory = train_tf(
                model,
                training_dataloader=train_dataloader,
                validation_dataloader=valid_dataloader,
            )
        elif framework == "pytorch1":
            print("train model pytorch1")
            mean_time, min_time, std_time, memory = train(
                model, 10, train_dataloader, valid_dataloader, device, "mean"
            )

        elif framework == "pytorch2":
            mean_time, min_time, std_time, memory = train(
                model, 10, train_dataloader, valid_dataloader, device, "mean"
            )
        elif framework == "pytorch2_compiled":
            res_time, memory = train(
                model, 1, train_dataloader, valid_dataloader, device
            )
    elif mode == "inference_gpu" or mode == "inference_cpu":
        if framework == "tf2":

            inference(
                model,
                test_dataloader=valid_dataloader,
                batch_size=batch_size,
                dataset_name=dataset_name,
                prep=prep,
            )
        elif framework == "pytorch1":
            mean_time, min_time, std_time, memory = inference(
                dataset_name=dataset_name,
                test_dataloader=valid_dataloader,
                device=device,
            )

        elif framework == "pytorch2":
            mean_time, min_time, std_time, memory = inference(
                dataset_name=dataset_name,
                test_dataloader=valid_dataloader,
                device=device,
            )
    else:
        raise ValueError(
            f'Wrong mode "{mode}". Must be "train", "inference_gpu" or "inference_cpu".'
        )
    print("training done")
    update_file(
        file_path,
        framework,
        input_shape,
        batch_size,
        mode,
        device,
        mean_time,
        min_time,
        std_time,
        memory,
        dataset_name,
        exc_info,
    )


def read_model_config(model_name: str) -> dict:
    with open("path_to_file/person.json", "r") as f:
        data = json.load(f)
    print(data[model_name])
    return data[model_name]


def update_file(
    file_path: str,
    framework: str,
    input_shape: Tuple[int, int, int],
    batch_size: int,
    mode: str,
    device: str,
    mean_time: str,
    min_time: str,
    std_time: str,
    memory: str,
    dataset_name: str,
    exc_info: str = "",
) -> None:
    """
    Updating (or creating new if not exists) log csv file with measured time and memory and other parameters.
    Creates directory if not exists. Printing logs to console.
    :param file_path: path to log csv file with measured time and memory.
    :param framework: "tf1", "tf2" or "pytorch".
    :param architecture: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
    :param input_shape: input image shape (height, width, channels).
    :param batch_size: batch size.
    :param mode: "train", "inference_gpu" or "inference_cpu".
    :param gpu_number: GPU index or None for using CPU.
    :param time: measured time.
    :param memory: measured GPU memory ("-" if CPU was used).
    :param exc_info: any exception info for printing.
    """
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        df = pd.DataFrame(
            columns=[
                "Framework",
                "Dataset",
                "Input size",
                "Batch size",
                "Mode",
                "Device",
                "Time_mean",
                "Time_min",
                "Time_std",
                "Memory",
            ]
        )
    else:
        df = pd.read_csv(file_path, sep=";", index_col=0)
    update_list = [
        framework,
        dataset_name,
        str(input_shape),
        batch_size,
        mode,
        device,
        mean_time,
        min_time,
        std_time,
        memory,
    ]
    df.loc[len(df)] = update_list
    df.to_csv(file_path, sep=";")

    cell_len = 20
    mode = mode.capitalize().replace("_", " ")
    input_shape = str(input_shape)
    batch_size = str(batch_size)
    # if time != "-":
    #     time = str(round(float(time), 6))

    # msg = f'{mode + " "*(cell_len - len(mode))}'
    # msg += f'{input_shape + " "*(cell_len - len(input_shape))}{batch_size + " "*(cell_len - len(batch_size))}'
    # msg += (
    #     f'{time + " "*(cell_len - len(time))}{memory + " "*(cell_len - len(memory))}'
    #     + exc_info
    # )
    # print(msg)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser(
        "Script for running single measurement of time and GPU memory using provided parameters."
    )
    parser.add_argument(
        "--file", type=str, help="Path to log csv file with measured time and memory."
    )
    parser.add_argument(
        "--framework",
        type=str,
        help='Framework: "tf1", "tf2" , "pytorch1" or "pytorch2".',
    )
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help='Work mode: "train", "inference_gpu" or "inference_cpu".',
    )
    parser.add_argument("--device", type=str, help='GPU index. Pass "-1" to use CPU.')
    parser.add_argument("--dataset", type=str, help="dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.framework.startswith("pytorch"):
        pytorch_version = get_pytorch_version()
        print(pytorch_version)
        if args.device == "mps":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                x = torch.ones(1, device=device)
                print(x)
            else:
                print("MPS device not found.")
        else:
            device = args.device
        logging.basicConfig(
            filename=f"results/{args.dataset}-torch-{pytorch_version}-{datetime.now()}.log",
            level=logging.DEBUG,
            format="%(asctime)s | %(message)s",
        )
    else:
        device = args.device
    print(device)
    single_test(
        file_path=args.file,
        framework=args.framework,
        batch_size=args.batch_size,
        mode=args.mode,
        device=device,
        dataset_name=args.dataset,
    )
