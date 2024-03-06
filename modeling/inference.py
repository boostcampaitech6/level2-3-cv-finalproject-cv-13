import os
import time
import numpy as np
import yaml
from glob import glob
from tqdm import tqdm

import torch

from dataloader import MRInferenceDataset

from sklearn import metrics

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

    return auc
        
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

    auc = test(model_name, test_loader)

    print(f"AUC : {auc}")
    t_end_training = time.time()
    print(f'Test took {t_end_training - t_start_training} s')

if __name__ == "__main__":
    for config_file in os.listdir('./configs'):
        with open(f'./configs/{config_file}', 'r') as file:
            config = yaml.safe_load(file)
        run(config)
