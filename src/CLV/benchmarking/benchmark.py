#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import random
import time
import sys
import logging
from datetime import datetime
from torch.utils.data import DataLoader

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)
print(root_folder)
import numpy as np
import torch
from pytorch_model.base_model import LSTMModel


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_classifier_simple_v2(
    model,
    num_epochs,
    train_loader,
    valid_loader,
    optimizer,
    device,
    logging_interval=50,
    best_model_save_path=None,
    scheduler=None,
    skip_train_acc=False,
    scheduler_on="valid_acc",
):

    start_time = time.time()
    logging.info(f"Start Time: {start_time:.2f}")
    minibatch_loss_list_train, minibatch_loss_list_val = [], []
    best_valid_loss, best_epoch = float("inf"), 0

    for epoch in range(num_epochs):

        epoch_start_time = time.time()
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
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
                    f"| Batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"| Loss: {loss:.4f}"
                )

        model.eval()

        elapsed = (time.time() - epoch_start_time) / 60
        logging.info(f"Time / epoch without evaluation: {elapsed:.2f} min")
        for i, vdata in enumerate(valid_loader):
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

    # test_acc = compute_accuracy(model, test_loader, device=device)
    # print(f"Test accuracy {test_acc :.2f}%")
    logging.info(
        f"Epoch: {epoch+1:03d}/{num_epochs:03d} "
        f"| Best Validation "
        f"(Ep. {best_epoch:03d}): {best_valid_loss :.2f}"
    )
    elapsed = (time.time() - start_time) / 60
    logging.info(f"Total Time: {elapsed:.2f} min")

    return minibatch_loss_list_train, minibatch_loss_list_val


def get_dataloaders_bank(batch_size: int, num_workers: int):

    train_dataset = torch.load("saved_datasets/bank_train_dataset.pt")

    valid_dataset = torch.load("saved_datasets/bank_valid_dataset.pt")

    # test_dataset = datasets.MNIST(root="data", train=False, transform=test_transforms)

    # if validation_fraction is not None:
    #     num = int(validation_fraction * 60000)
    #     train_indices = torch.arange(0, 60000 - num)
    #     valid_indices = torch.arange(60000 - num, 60000)

    #     train_sampler = SubsetRandomSampler(train_indices)
    #     valid_sampler = SubsetRandomSampler(valid_indices)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # else:
    #     train_loader = DataLoader(
    #         dataset=train_dataset,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         drop_last=True,
    #         shuffle=True,
    #     )

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     shuffle=False,
    # )

    # if validation_fraction is None:
    #     return train_loader, test_loader
    # else:
    #     return train_loader, valid_loader, test_loader
    return train_loader, valid_loader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, required=True, help="Which GPU device to use."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Which dataset to use."
    )

    args = parser.parse_args()

    RANDOM_SEED = 123
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    DEVICE = torch.device(args.device)
    DATASET = args.dataset
    print("torch", torch.__version__)
    print("device", DEVICE)
    logging.basicConfig(
        filename=f"results/{DATASET}-torch-{torch.__version__}-{datetime.now()}.log",
        level=logging.DEBUG,
        format="%(asctime)s | %(message)s",
    )
    train_loader, valid_loader = get_dataloaders_bank(
        batch_size=BATCH_SIZE, num_workers=0
    )

    torch.manual_seed(RANDOM_SEED)

    model = LSTMModel(max_weeks=52, max_trans=12, hidden_size=128, stateful=False)

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    minibatch_loss_list_train, minibatch_loss_list_val = train_classifier_simple_v2(
        model=model,
        num_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        best_model_save_path=None,
        device=DEVICE,
        skip_train_acc=True,
        scheduler_on="valid_acc",
        logging_interval=100,
    )
