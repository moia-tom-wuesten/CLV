import os, sys
from torch.utils.data import TensorDataset, DataLoader
import psutil
import torch
import logging
import time

# import yappi
import numpy as np
from tqdm import trange


import json


root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)

from pytorch_model.base_model import LSTMModel


def get_dataloaders_bank(dataset_name: str, batch_size: int, num_workers: int):
    x_train = torch.load(f"saved_datasets/{dataset_name}_x_train.pt")
    x_valid = torch.load(f"saved_datasets/{dataset_name}_x_valid.pt")
    y_train = torch.load(f"saved_datasets/{dataset_name}_y_train.pt")
    y_valid = torch.load(f"saved_datasets/{dataset_name}_y_valid.pt")
    train_df = TensorDataset(x_train, y_train)
    valid_df = TensorDataset(x_valid, y_valid)
    valid_loader = DataLoader(
        dataset=valid_df,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        dataset=train_df,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_loader, valid_loader


def load_pytorch_model(dataset_name, device):
    with open("feature_config.json", "r") as f:
        data = json.load(f)
    max_weeks = data[dataset_name]["features"]["weeks"]
    max_trans = data[dataset_name]["features"]["trans"]
    model = LSTMModel(
        max_weeks=max_weeks,
        max_trans=max_trans,
        stateful=False,
        hidden_size=128,
        device=device,
    )
    return model


def get_pytorch_version():
    return torch.__version__


def train(
    model,
    num_epochs,
    train_dataloader,
    valid_dataloader,
    device,
    measure_time_mode,
    logging_interval=50,
    best_model_save_path=None,
    scheduler=None,
    skip_train_acc=False,
    scheduler_on="valid_acc",
):
    list_of_cpu_usage = list()
    minibatch_loss_list_train, minibatch_loss_list_val = [], []
    best_valid_loss, best_epoch = float("inf"), 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    list_of_performance_times = []
    for epoch in range(num_epochs):

        epoch_start_time = time.time()
        # yappi.set_clock_type("cpu")
        # yappi.start()
        model.train()
        for batch_idx, (features, targets) in enumerate(train_dataloader):
            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            list_of_cpu_usage.append(psutil.cpu_percent())
            loss = torch.nn.functional.cross_entropy(
                logits.permute(0, 2, 1), targets.squeeze()
            )
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list_train.append(loss.item())
            if not batch_idx % logging_interval:
                logging.info(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
                    f"| Batch {batch_idx:04d}/{len(train_dataloader):04d} "
                    f"| Loss: {loss:.4f}"
                )

        model.eval()
        for i, vdata in enumerate(valid_dataloader):
            with torch.no_grad():  # save memory during inference
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device=device)
                vlabels = vlabels.to(device)
                valY_pred = model(vinputs)
                # val_loss=loss_function(valY_pred,torch.max(vlabels, dim=0)[0])
                val_los = torch.nn.functional.cross_entropy(
                    valY_pred.permute(0, 2, 1), vlabels.squeeze()
                )
                minibatch_loss_list_val.append(val_los.item())

            if val_los < best_valid_loss:
                best_valid_loss, best_epoch = val_los, epoch + 1
                if best_model_save_path:
                    torch.save(model.state_dict(), best_model_save_path)
        # yappi.get_func_stats().print_all()
        elapsed = time.time() - epoch_start_time
        list_of_performance_times.append(elapsed)
    mean_elapsed = np.mean(list_of_performance_times)
    min_elapsed = np.min(list_of_performance_times)
    std_elapsed = np.std(list_of_performance_times)
    print(f"{min_elapsed=}, {mean_elapsed=}, {std_elapsed=}")
    logging.info(f"Total Training Time: {mean_elapsed:.2f} min")
    cpu_usage = sum(list_of_cpu_usage) / len(list_of_cpu_usage)
    # test_acc = compute_accuracy(model, test_loader, device=device)
    # print(f"Test accuracy {test_acc :.2f}%")
    logging.info(
        f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
        f"| Best Validation "
        f"(Ep. {best_epoch:03d}): {best_valid_loss :.2f}"
    )
    return str(mean_elapsed), str(min_elapsed), str(std_elapsed), str(cpu_usage)


def inference(dataset_name: str, test_dataloader: DataLoader, device: str):
    """
    Measure inference step time and GPU memory using provided model with given parameters.
    :param model_name: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
    :param batch_size: batch size.
    :param input_shape: input image shape (height, width, channels).
    :return: measured mean inference step time and GPU memory.
    """
    torch.cuda.empty_cache()

    model = load_pytorch_model(dataset_name=dataset_name, device=device)
    model = model.to(device)
    model.eval()
    list_of_performance_times = []
    for i, vdata in enumerate(test_dataloader):
        with torch.no_grad():
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device=device)
            vlabels = vlabels.to(device)
            for step_idx in range(10):
                start_time = time.time()
                valY_pred = model(vinputs)
                finish_time = time.time()
                list_of_performance_times.append(finish_time - start_time)
    mean_elapsed = np.mean(list_of_performance_times)
    min_elapsed = np.min(list_of_performance_times)
    std_elapsed = np.std(list_of_performance_times)
    return str(mean_elapsed), str(min_elapsed), str(std_elapsed), ""


if __name__ == "__main__":
    # train_dataloader, valid_dataloader = get_dataloaders_bank("bank", 62, 0)
    # model = load_pytorch_model("bank")
    # compiled_model = torch.compile(model)
    # print(compiled_model)
    # elapsed, cpu_time = train(
    #     model=compiled_model,
    #     num_epochs=10,
    #     train_dataloader=train_dataloader,
    #     valid_dataloader=valid_dataloader,
    #     device="cpu",
    # )
    get_pytorch_version()

    def foo(x, y):
        a = torch.sin(x)
        b = torch.cos(x)
        return a + b

    opt_foo1 = torch.compile(foo)
    print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))
