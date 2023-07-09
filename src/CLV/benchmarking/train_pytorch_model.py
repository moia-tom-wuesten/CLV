import os, sys
from torch.profiler import profile, record_function, ProfilerActivity
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
        # with profile(
        #     activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
        # ) as prof:
        model.train()
        for batch_idx, (features, targets) in enumerate(train_dataloader):
            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            # list_of_cpu_usage.append(psutil.gp())
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
    # test_acc = compute_accuracy(model, test_loader, device=device)
    # print(f"Test accuracy {test_acc :.2f}%")
    logging.info(
        f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
        f"| Best Validation "
        f"(Ep. {best_epoch:03d}): {best_valid_loss :.2f}"
    )
    # prof.export_chrome_trace("trace.json")
    return (
        str(mean_elapsed),
        str(min_elapsed),
        str(std_elapsed),
        list_of_cpu_usage,
        list_of_performance_times,
    )


def train_profile(model, data, device):
    model.train()
    features, targets = data[0].to(device=device), data[1].to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # ## FORWARD AND BACK PROP
    logits = model(features)
    loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), targets.squeeze())
    optimizer.zero_grad()

    loss.backward()

    # ## UPDATE MODEL PARAMETERS
    optimizer.step()


def eval_profile(model, data, device):
    model.eval()
    vinputs, vlabels = data
    with torch.no_grad():  # save memory during inference
        vinputs = vinputs.to(device=device)
        vlabels = vlabels.to(device)
        valY_pred = model(vinputs)
        # val_loss=loss_function(valY_pred,torch.max(vlabels, dim=0)[0])
        val_los = torch.nn.functional.cross_entropy(
            valY_pred.permute(0, 2, 1), vlabels.squeeze()
        )


# def profile(
#     model,
#     train_dataloader,
#     validation_dataloader,
#     device,
#     logging_interval=50,
#     best_model_save_path=None,
#     scheduler=None,
#     skip_train_acc=False,
#     scheduler_on="valid_acc",
# ):
#     prof = torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/pytorch1_eval"),
#         record_shapes=True,
#         with_stack=True,
#     )
#     prof.start()
#     for step, batch_data in enumerate(train_dataloader):
#         if step >= (1 + 1 + 3) * 2:
#             break
#         train_profile(model, batch_data, device)
#     for step, batch_data in enumerate(validation_dataloader):
#         if step >= (1 + 1 + 3) * 2:
#             break
#         eval_profile(model, batch_data, device)
#         prof.step()
#     prof.stop()


def profile2(
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
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/new_pytroch"),
        record_shapes=True,
        with_stack=True,
    )
    prof.start()
    epoch = 0
    list_of_cpu_usage = list()
    minibatch_loss_list_train, minibatch_loss_list_val = [], []
    best_valid_loss, best_epoch = float("inf"), 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    list_of_performance_times = []
    epoch_start_time = time.time()
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
    prof.stop()


def inference_torch(dataset_name: str, test_dataloader: DataLoader, device: str):
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
    list_of_cpu_usage = list()
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
    return (
        str(mean_elapsed),
        str(min_elapsed),
        str(std_elapsed),
        "",
        list_of_performance_times,
    )


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
    import torch

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")
