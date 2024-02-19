import os
import time
import numpy as np
import yaml
import wandb

import torch
import torch.optim as optim

from dataloader import MRDataset
import model

from sklearn import metrics


def train_model(model, train_loader, epoch, num_epochs, optimizer, current_lr):
    _ = model.train()

    if torch.cuda.is_available():
        model.cuda()

    y_preds = []
    y_trues = []
    losses = []

    for image, label, weight in train_loader:
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float())

        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][1]))
        y_preds.append(probas[0][1].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    
    print(f"[Epoch: {epoch} / {num_epochs} | "
          f"train loss {train_loss_epoch} | train auc : {train_auc_epoch} | lr : {current_lr}]")
    
    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, epoch, num_epochs, current_lr):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []

    for image, label, weight in val_loader:
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float())

        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][1]))
        y_preds.append(probas[0][1].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)

    print(f"[Epoch: {epoch} / {num_epochs} | "
        f"valid loss {val_loss_epoch} | valid auc : {val_auc_epoch} | lr : {current_lr}]")
        
    return val_loss_epoch, val_auc_epoch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run(config):
    DATA_ROOT = config['DATA_ROOT']
    
    EXP_NAME = config['EXP_NAME']
    CAMPER_ID = config['CAMPER_ID']

    TASK = config['TASK']
    PLANE = config['PLANE']
    
    NUM_EPOCHS = config['epochs']
    
    wandb.init(project='Boost Camp Lv3', entity='frostings', name=f"{CAMPER_ID}-{EXP_NAME}", config=config)

    train_dataset = MRDataset(DATA_ROOT, TASK, PLANE, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=8, drop_last=False)

    validation_dataset = MRDataset(DATA_ROOT, TASK, PLANE, train=False)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=-True, num_workers=8, drop_last=False)

    mrnet = model.MRNet()

    if torch.cuda.is_available():
        mrnet = mrnet.cuda()

    optimizer = optim.Adam(mrnet.parameters(), lr=0.00001, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)

    best_val_loss = float('inf')
    best_val_auc = float(0)

    t_start_training = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = get_lr(optimizer)

        t_start = time.time()
        
        train_loss, train_auc = train_model(
            mrnet, train_loader, epoch, NUM_EPOCHS, optimizer, current_lr)
        val_loss, val_auc = evaluate_model(
            mrnet, validation_loader, epoch, NUM_EPOCHS, current_lr)
        
        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - t_start

        print(f"train loss : {train_loss} | train auc {train_auc} | "
              f"val loss {val_loss} | val auc {val_auc} | elapsed time {delta} s")
        
        wandb.log({"train/loss": train_loss, "train/auc": train_auc}, step=epoch)
        wandb.log({"valid/loss": val_loss, "valid/auc": val_auc}, step=epoch)

        print('-' * 50)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            file_name = f'model_{EXP_NAME}_{TASK}_{PLANE}_epoch_{epoch}.pth'
            for f in os.listdir('./models/'):
                if (EXP_NAME in f) and (TASK in f) and (PLANE in f):
                    os.remove(f'./models/{f}')
            torch.save(mrnet, f'./models/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')
    wandb.finish()


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    run(config)
