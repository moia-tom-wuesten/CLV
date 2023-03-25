import os, sys
from torch.utils.data import TensorDataset, DataLoader
import psutil
import torch
import logging
import time
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


def load_pytorch_model(dataset_name):
    with open("feature_config.json", "r") as f:
        data = json.load(f)
    max_weeks = data[dataset_name]["features"]["weeks"]
    max_trans = data[dataset_name]["features"]["trans"]
    model = LSTMModel(
        max_weeks=max_weeks, max_trans=max_trans, stateful=False, hidden_size=128
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
    logging_interval=50,
    best_model_save_path=None,
    scheduler=None,
    skip_train_acc=False,
    scheduler_on="valid_acc",
):
    list_of_cpu_usage = list()
    start_time = time.time()
    logging.info(f"Start Time: {start_time:.2f}")
    minibatch_loss_list_train, minibatch_loss_list_val = [], []
    best_valid_loss, best_epoch = float("inf"), 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):

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

        elapsed = (time.time() - epoch_start_time) / 60
        logging.info(f"Time / epoch without evaluation: {elapsed:.2f} min")
        for i, vdata in enumerate(valid_dataloader):
            with torch.no_grad():  # save memory during inference
                vinputs, vlabels = vdata
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

        elapsed = (time.time() - start_time) / 60

        logging.info(f"Time elapsed: {elapsed:.2f} min")

        if scheduler is not None:

            if scheduler_on == "valid_acc":
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == "minibatch_loss":
                scheduler.step(minibatch_loss_list_train[-1])
            else:
                raise ValueError("Invalid `scheduler_on` choice.")

    elapsed = (time.time() - start_time) / 60
    logging.info(f"Total Training Time: {elapsed:.2f} min")
    cpu_usage = sum(list_of_cpu_usage) / len(list_of_cpu_usage)
    # test_acc = compute_accuracy(model, test_loader, device=device)
    # print(f"Test accuracy {test_acc :.2f}%")
    logging.info(
        f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
        f"| Best Validation "
        f"(Ep. {best_epoch:03d}): {best_valid_loss :.2f}"
    )
    elapsed = (time.time() - start_time) / 60
    print(elapsed)
    logging.info(f"Total Time: {elapsed:.2f} min")
    return str(elapsed), str(cpu_usage)


def inference(batch_size: int, dataset_name: str):
    """
    Measure inference step time and GPU memory using provided model with given parameters.
    :param model_name: one of the available architectures (see tf_torch_models.utils.AVAILABLE_MODELS).
    :param batch_size: batch size.
    :param input_shape: input image shape (height, width, channels).
    :return: measured mean inference step time and GPU memory.
    """
    torch.cuda.empty_cache()

    x_inference = torch.load(f"saved_datasets/{dataset_name}_x_train.pt")

    model = load_pytorch_model(dataset_name=dataset_name)
    model.eval()

    desc_str = (
        f'Inference: Architecture: "{torch.__version__}". Batch size: "{batch_size}".'
    )
    all_times = []

    with torch.no_grad():
        for step_idx in trange(10, leave=False, desc=desc_str):
            batch_start = step_idx * batch_size
            batch_end = (step_idx + 1) * batch_size
            start_time = time.time()
            batch = torch.tensor(x_inference[batch_start:batch_end, :, :]).long()
            model(batch)
            finish_time = time.time()
            all_times.append(finish_time - start_time)

    mean_time = sum(all_times) / len(all_times)
    return str(mean_time), ""
