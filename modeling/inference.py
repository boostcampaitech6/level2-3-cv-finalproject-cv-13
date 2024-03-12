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

column_list=['AUC', 'Accuracy', 'F1-score', 'Precision', 'Recall', 'Specificity']
df = pd.DataFrame(columns=column_list)
# df.loc['abnormal-axial'] = [1,2,3,4,5,6]
# print(df)


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
        
def run(config):
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
    print(f"| AUC : {test_metric['auc']} | Accuracy : {test_metric['acc']} | F1-score : {test_metric['f1']} |")
    print(f"| Precision : {test_metric['precision']} | Recall : {test_metric['recall']} | Specificity : {test_metric['specificity']} |")
    print('-' * 150)
    df.loc[f'{TASK}-{PLANE}'] = [test_metric['auc'], test_metric['acc'], test_metric['f1'],
                                  test_metric['precision'], test_metric['recall'], test_metric['specificity']]
    t_end_training = time.time()
    print(f'Test took {t_end_training - t_start_training} s')

if __name__ == "__main__":
    for config_file in sorted(os.listdir('./configs')):
        with open(f'./configs/{config_file}', 'r') as file:
            config = yaml.safe_load(file)
        run(config)
    df.to_csv('result.csv')
