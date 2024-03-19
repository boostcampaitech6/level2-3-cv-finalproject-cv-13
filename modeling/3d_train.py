import os
import time
import numpy as np
import yaml
import wandb
import random

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms

from dataloader import MRDataset
from metric import Metric
import model
from loss import create_criterion
from optimizer import create_optim
from scheduler import create_sched
from model import create_model

from sklearn import metrics
from torchvision import transforms

import code

seed = 2024
deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, epoch, num_epochs, LOSS, optimizer, current_lr):
    _ = model.train()

    # loss 정의
    loss_name = LOSS['name']
    loss_params = LOSS['params'] or {}

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

        prediction = model.forward(image.float())

        criterion = create_criterion(loss_name, weight, **loss_params)
        loss = criterion(prediction, label)

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        for i in range(label.shape[0]):
            y_trues.append(int(label[i][1]))
            y_preds.append(probas[i][1].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5
        break
            
    y_preds = [1 if y_pred > 0.5 else 0 for y_pred in y_preds]
    metric = Metric()
    metric(torch.tensor(y_trues), torch.tensor(y_preds))
    metric_result = metric.update()
    metric_result['loss'] = np.round(np.mean(losses), 4)
    metric_result['auc'] = np.round(auc, 4)
    
    print(f"[Epoch: {epoch} / {num_epochs} | "
          f"train loss {metric_result['loss']} | train auc : {metric_result['auc']} | lr : {current_lr}]")
    
    
    ##########################################
    cm = metrics.confusion_matrix(y_trues, y_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ["0", "1"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    title = f"{config['TASK']}_{config['PLANE']}"
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    
    image_path = "confusion_matrix.png"
    plt.savefig(image_path)
    wandb.log({f"train_confusion_matrix": wandb.Image(image_path)}, step=epoch)
    ##########################################
    
    return metric_result


def evaluate_model(model, val_loader, epoch, num_epochs, LOSS, current_lr):
    _ = model.eval()

    # loss 정의
    loss_name = LOSS['name']
    loss_params = LOSS['params'] or {}
    

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

        prediction = model.forward(image.float())

        criterion = create_criterion(loss_name, weight, **loss_params)
        loss = criterion(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        for i in range(label.shape[0]):
            y_trues.append(int(label[i][1]))
            y_preds.append(probas[i][1].item())

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5
        break
            
    y_preds = [1 if y_pred > 0.5 else 0 for y_pred in y_preds]
    metric = Metric()
    metric(torch.tensor(y_trues), torch.tensor(y_preds))
    metric_result = metric.update()
    metric_result['loss'] = np.round(np.mean(losses), 4)
    metric_result['auc'] = np.round(auc, 4)

    print(f"[Epoch: {epoch} / {num_epochs} | "
        f"valid loss {metric_result['loss']} | valid auc : {metric_result['auc']} | lr : {current_lr}]")
    
    ##########################################
    cm = metrics.confusion_matrix(y_trues, y_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ["0", "1"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    title = f"{config['TASK']}_{config['PLANE']}"
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    
    image_path = "confusion_matrix.png"
    plt.savefig(image_path)
    wandb.log({f"eval_confusion_matrix": wandb.Image(image_path)}, step=epoch)
    ##########################################
    
    return metric_result


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def pad_sequences_3d_uniform(sequence, max_depth=61):
    depth_padding = max_depth - sequence.shape[0]
    pad_before = depth_padding // 2
    pad_after = depth_padding - pad_before
    pad = ((pad_before, pad_after), (0, 0), (0, 0))
    padded_sequence = np.pad(sequence, pad, 'constant', constant_values=0)
    return torch.tensor(padded_sequence).float()


def run(config):
    DATA_ROOT = config['DATA_ROOT']
    
    EXP_NAME = config['EXP_NAME']
    CAMPER_ID = config['CAMPER_ID']

    TASK = config['TASK']
    PLANE = config['PLANE']
    
    NUM_EPOCHS = config['epochs']
    LR = config['LR']
    BATCH_SIZE = config['BATCH_SIZE']
    FOLD_NUM = config['FOLD_NUM']

    OPTIMIZER = config['OPTIMIZER']
    LOSS = config['LOSS']
    SCHEDULER = config['SCHEDULER']
    MODEL = config['MODEL']
    
    wandb.init(project='Boost Camp Lv3', entity='frostings', name=f"{CAMPER_ID}-{EXP_NAME}-{TASK}-{PLANE}", config=config)

    train_augmentor = transforms.Compose([
        transforms.Lambda(lambda x: pad_sequences_3d_uniform(x)),
        transforms.Lambda(lambda x: torch.Tensor(x)),
        transforms.RandomRotation(25),
        transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1)),
    ])

    valid_augmentor = transforms.Compose([
        transforms.Lambda(lambda x: pad_sequences_3d_uniform(x)),
        transforms.Lambda(lambda x: torch.Tensor(x)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1))
    ])

    train_dataset = MRDataset(DATA_ROOT, TASK, PLANE, FOLD_NUM, train=True, transform=train_augmentor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=False)

    validation_dataset = MRDataset(DATA_ROOT, TASK, PLANE, FOLD_NUM, train=False, transform=valid_augmentor)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=-True, num_workers=8, drop_last=False)

    print(len(train_dataset), len(validation_dataset))

    model_name = MODEL['name']
    model_params = MODEL['params'] or {}
    mrnet = create_model(model_name, **model_params)

    if torch.cuda.is_available():
        mrnet = mrnet.cuda()

    # optimizer 정의
    optimizer_name = OPTIMIZER['name']
    optimizer_params = OPTIMIZER['params'] or {}
    optimizer = create_optim(optimizer_name, mrnet, LR, **optimizer_params)

    # scheduler 정의
    scheduler_name = SCHEDULER['name']
    scheduler_params = SCHEDULER['params'] or {}

    scheduler = None
    is_plateau = False
    if scheduler_name != "":
        scheduler, is_plateau = create_sched(scheduler_name, optimizer, NUM_EPOCHS, **scheduler_params)

    best_val_loss = float('inf')
    best_val_auc = float(0)

    t_start_training = time.time()
    os.makedirs('models', exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = get_lr(optimizer)

        t_start = time.time()
        
        train_metric = train_model(
            mrnet, train_loader, epoch, NUM_EPOCHS, LOSS, optimizer, current_lr)
        val_metric = evaluate_model(
            mrnet, validation_loader, epoch, NUM_EPOCHS, LOSS, current_lr)
        
        if scheduler:
            if is_plateau:
                scheduler.step(val_metric['loss'])
            else:
                scheduler.step()

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

            if not os.path.exists('models'):                                                           
                os.makedirs('models')
    
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
