import time
import os
import sys
import copy
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

from sklearn.metrics import f1_score

# from tqdm import tqdm

import matplotlib.pyplot as plt


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
    device="cpu",
    model_dir="models/",
):

    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)

    print(f"model on device {device}, training for {num_epochs} epochs")
    print(f"Save at {str(model_dir)}")
    since = time.time()

    SAVE_BEST = True
    STORE_PREDS = False

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    track_learning = {
        "trn": {"loss": [], "accuracy": []},
        "val": {"loss": [], "accuracy": []},
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        sched = False
        # Each epoch has a training and validation phase
        for phase in ["trn", "val"]:
            n_batches = len(dataloaders[phase])
            batch_size = dataloaders[phase].batch_size
            n_data = batch_size * n_batches

            if phase == "trn":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if STORE_PREDS:
                yhat, gt, probs = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

            # Iterate over data.
            n_cur = 0
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                n_cur += len(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "trn"):
                    outputs = model(inputs)
                    proba = nn.Softmax(dim=1)(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # statistics
                    running_corrects += torch.sum(preds == labels.data)

                    # backward + optimize only if in training phase
                    if phase == "trn":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()  # * inputs.size(0)
                    # if phase == "val":
                    #     running_loss += loss.item()  # * inputs.size(0)

                if STORE_PREDS:
                    yhat = torch.cat((yhat, preds.cpu()))
                    gt = torch.cat((gt, labels.cpu()))
                    probs = torch.cat((probs, proba.cpu()))

                print(
                    f"\r---> {100*n_cur/n_data:.4f}% ({n_cur}/{n_data}) :: Loss {running_loss/(i + 1):.4f}, Acc {100*running_corrects/n_cur:.4f}",
                    end="\r",
                )

            if phase == "val":
                running_val_loss = running_loss
                sched = True

            if (phase == "trn") and sched:
                scheduler.step(running_val_loss)

            epoch_loss = running_loss / n_batches
            epoch_acc = running_corrects.cpu().numpy() / n_data
            track_learning[phase]["loss"].append(epoch_loss)
            track_learning[phase]["accuracy"].append(epoch_acc)

            print(f"\r\r ", end="\n")
            print(
                f"\r{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:2.2f}", end="\n"
            )

            # deep copy the model
            if phase == "val":
                np.save(f"{model_dir}/learning_curves.dict", track_learning)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model.state_dict())

                    if SAVE_BEST:
                        torch.save(
                            model.state_dict(), f"{model_dir}/model_state_best.pt"
                        )
                        torch.save(model, f"{model_dir}/model_pars_best.pt")

                # torch.save(model.state_dict(), f"{model_dir}/model_state_{epoch}.pt")
                # torch.save(model, f"{model_dir}/model_pars_{epoch}.pt")

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model)

    return model, track_learning


def infer_model(model, dataloader, criterion, device="cpu"):

    # print(f"model on device {device}, inference on {dataloader.dataset.learning_set}")
    since = time.time()

    # Each epoch has a training and validation phase
    n_data = dataloader.batch_size * len(dataloader)

    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    yhat, gt, probs = torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

    # Iterate over data.
    n_cur = 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        n_cur += len(labels)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            proba = nn.Softmax(dim=1)(outputs)
            _, preds = torch.max(outputs, 1)

            if criterion is not None:
                loss = criterion(outputs, labels)
            else:
                loss = torch.Tensor([0])

        # statistics
        running_loss += loss.item()  # * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # track preds
        yhat = torch.cat((yhat, preds.cpu()))
        gt = torch.cat((gt, labels.cpu()))
        probs = torch.cat((probs, proba.cpu()))
        # running_preds_gt.append([])

        print(
            f"\r\r---> {100*n_cur/n_data:.4f} ({n_cur}/{n_data}) :: Loss {running_loss/(i + 1):.4f}, Acc {100*running_corrects/n_cur:.4f}",
            end="\r",
        )

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / n_data
    # epoch_f1 =
    # print("\r\r")
    print(
        f"Inference Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1 {f1_score(gt, yhat):.4f}"
    )

    time_elapsed = time.time() - since
    print(f"Inference complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    return yhat, probs, gt


def visualize_model(
    model,
    dataloader,
    denormalize=None,
    num_images=6,
    device="cpu",
    figsize=(5, 5),
    save_folder="./models/figures/",
):

    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    model.eval()
    images_so_far = 0

    class_names = ["No bird", "Bird"]

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1

                fig, ax = plt.subplots(1, 1, figsize=figsize)

                # ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"P: {class_names[preds[j]]}, T: {class_names[labels[j]]}")

                im = inputs.cpu().data[j]
                if denormalize:
                    im = denormalize(im)

                im = im.permute((1, 2, 0))

                ax.imshow(im)

                fig.savefig(f"{save_folder}/fig_b{i}_im{j}.jpg")

                if images_so_far == num_images:
                    return
