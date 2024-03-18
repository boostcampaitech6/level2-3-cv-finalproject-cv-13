import os
import time
import numpy as np
import yaml
from glob import glob
from tqdm import tqdm
import pandas as pd

import torch

from dataloader import MRInferenceDataset
from metric import Metric

from sklearn import metrics

column_list=['Task', 'Plane', 'AUC', 'Accuracy', 'F1-score', 'Precision', 'Recall', 'Specificity', 'Exp_name']

def test(model_name, data_loader):
    model = torch.load(model_name)
    _ = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    y_trues = []
    y_preds = []
    
    for step, (image, label, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        label = label[0]

        prediction = model.forward(image.float()) 
        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][1]))
        y_preds.append(probas[0][1].item())      

    auc = np.round(metrics.roc_auc_score(y_trues, y_preds), 4)
    y_preds = [1 if y_pred > 0.5 else 0 for y_pred in y_preds]

    
    metric = Metric()
    metric(torch.tensor(y_trues), torch.tensor(y_preds))
    metric_result = metric.update()
    metric_result['auc'] = auc

    return metric_result
        
def run(config, df):
    DATA_ROOT = config['DATA_ROOT']
    MODEL_ROOT = config['MODEL_ROOT']
    EXP_NAME = config['EXP_NAME']
    BATCH_SIZE = config['BATCH_SIZE']

    TASK = config['TASK']
    PLANE = config['PLANE']

    
    model_name = glob(os.path.join(MODEL_ROOT, f'*_{EXP_NAME}_{TASK}_{PLANE}_*.pth'))[0]
    print(model_name)
    test_dataset = MRInferenceDataset(DATA_ROOT, TASK, PLANE)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)

    print(len(test_dataset))

    t_start_training = time.time()

    test_metric = test(model_name, test_loader)
    
    print('-' * 150)
    print(f"| AUC : {np.round(test_metric['auc'], 4)} | Accuracy : {np.round(test_metric['acc'], 4)} | F1-score : {np.round(test_metric['f1'], 4)} |")
    print(f"| Precision : {np.round(test_metric['precision'], 4)} | Recall : {np.round(test_metric['recall'], 4)} | Specificity : {np.round(test_metric['specificity'], 4)} |")
    print('-' * 150)

    data = [TASK, PLANE, np.round(test_metric['auc'], 4), np.round(test_metric['acc'], 4), np.round(test_metric['f1'], 4),
            np.round(test_metric['precision'], 4), np.round(test_metric['recall'], 4), np.round(test_metric['specificity'], 4), EXP_NAME]
    tmp = pd.DataFrame([data], columns=column_list)
    df = pd.concat([df, tmp], axis=0)

    t_end_training = time.time()
    print(f'Test took {t_end_training - t_start_training} s')
    return df

if __name__ == "__main__":
    df = pd.DataFrame(columns=column_list)

    for config_file in sorted(os.listdir('./configs')):
        with open(f'./configs/{config_file}', 'r') as file:
            config = yaml.safe_load(file)
        df = run(config, df)
        
    df = pd.pivot_table(df, index=['Exp_name', 'Task', 'Plane'], columns=[])
    df.to_csv('result.csv')
