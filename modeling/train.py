import os
import time
import numpy as np
import yaml
import wandb

import torch
import torch.optim as optim

from dataloader import MRDataset
from metric import Metric
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
            
    y_preds = [1 if y_pred > 0.5 else 0 for y_pred in y_preds]
    metric = Metric()
    metric(torch.tensor(y_trues), torch.tensor(y_preds))
    metric_result = metric.update()
    metric_result['loss'] = np.round(np.mean(losses), 4)
    metric_result['auc'] = np.round(auc, 4)
    
    print(f"[Epoch: {epoch} / {num_epochs} | "
          f"train loss {metric_result['loss']} | train auc : {metric_result['auc']} | lr : {current_lr}]")
    
    return metric_result


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
            
    y_preds = [1 if y_pred > 0.5 else 0 for y_pred in y_preds]
    metric = Metric()
    metric(torch.tensor(y_trues), torch.tensor(y_preds))
    metric_result = metric.update()
    metric_result['loss'] = np.round(np.mean(losses), 4)
    metric_result['auc'] = np.round(auc, 4)

    print(f"[Epoch: {epoch} / {num_epochs} | "
        f"valid loss {metric_result['loss']} | valid auc : {metric_result['auc']} | lr : {current_lr}]")
        
    return metric_result


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
    os.makedirs('models', exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = get_lr(optimizer)

        t_start = time.time()
        
        train_metric = train_model(
            mrnet, train_loader, epoch, NUM_EPOCHS, optimizer, current_lr)
        val_metric = evaluate_model(
            mrnet, validation_loader, epoch, NUM_EPOCHS, current_lr)
        
        scheduler.step(val_metric['loss'])

        t_end = time.time()
        delta = t_end - t_start

        print(f"train loss : {train_metric['loss']} | train auc {train_metric['auc']} | "
              f"val loss {val_metric['loss']} | val auc {val_metric['auc']} | elapsed time {delta} s")
        
        
        for k, v in train_metric.items():
            wandb.log({f"train/{k}": v}, step=epoch)
        
        for k, v in val_metric.items():
            wandb.log({f"valid/{k}": v}, step=epoch)

        print('-' * 50)
        
        if val_metric['auc'] > best_val_auc:
            best_val_auc = val_metric['auc']
            file_name = f'model_{EXP_NAME}_{TASK}_{PLANE}_epoch_{epoch}.pth'
            for f in os.listdir('./models/'):
                if (EXP_NAME in f) and (TASK in f) and (PLANE in f):
                    os.remove(f'./models/{f}')
            torch.save(mrnet, f'./models/{file_name}')

        if val_metric['loss'] < best_val_loss:
            best_val_loss = val_metric['loss']

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')
    wandb.finish()


if __name__ == "__main__":
    for config_file in os.listdir('./configs'):
        with open(f'./configs/{config_file}', 'r') as file:
            config = yaml.safe_load(file)
        run(config)
