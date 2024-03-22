import torch
from sklearn.linear_model import LogisticRegression
import yaml
import code
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score

from dataloader import MRDataset, MRInferenceDataset
from metric import Metric


class FusionModel:
    def __init__(self, data_root, task, model_path, fold_num, exp_name='temp'):
        self.data_root = data_root
        self.task = task
        self.model_path = model_path
        self.exp_name = exp_name
        self.fold_num = fold_num
        self.logistic_regression = LogisticRegression()

    def build_dataset(self):
        os.makedirs(f'preds/{self.exp_name}', exist_ok=True)

        for plane in ['axial', 'coronal', 'sagittal']:
            model = torch.load(self.model_path[plane])
            _ = model.eval()
            if torch.cuda.is_available():
                model.cuda()
            
            for data_type in ['train', 'test']:
                if data_type == 'train':
                    dataset = MRDataset(self.data_root, self.task, plane, self.fold_num, train=True)
                else:
                    dataset = MRInferenceDataset(self.data_root, self.task, plane)

                data = []
                for i, (array, _, _) in enumerate(dataset):
                    if torch.cuda.is_available():
                        array = array.cuda()
                    
                    subject_id = dataset.ids[i]
                    prediction = model.forward(array.float())
                    probas = torch.sigmoid(prediction)[0][1].item()

                    data.append([subject_id, probas])
                    
                    if i % 100 == 0 and i > 0:
                        print(f"[{data_type}] {plane}: {i}/{len(dataset)}")
                
                df = pd.DataFrame(data)
                df.to_csv(f'./preds/{self.exp_name}/{data_type}-{plane}.csv', index=False, header=False)
            print(f'\n{plane.capitalize()} Building, End!\n')
    
    def fit(self):
        axial_df = pd.read_csv(f'./preds/{self.exp_name}/train-axial.csv', 
                               header=None, names=['id', 'axial'])
        coronal_df = pd.read_csv(f'./preds/{self.exp_name}/train-coronal.csv', 
                                 header=None, names=['id', 'coronal'])
        sagittal_df = pd.read_csv(f'./preds/{self.exp_name}/train-sagittal.csv', 
                                  header=None, names=['id', 'sagittal'])
        label_df = pd.read_csv(os.path.join(self.data_root, f'train-{self.task}.csv'), 
                               header=None, names=['id', 'label'])

        merged_df = axial_df.merge(coronal_df, on='id').merge(sagittal_df, on='id').merge(label_df, on='id')

        X = merged_df[['axial', 'coronal', 'sagittal']]
        y = merged_df['label']

        self.logistic_regression.fit(X, y)
    
    def evaluate(self):
        axial_df = pd.read_csv(f'./preds/{self.exp_name}/test-axial.csv', 
                        header=None, names=['id', 'axial'])
        coronal_df = pd.read_csv(f'./preds/{self.exp_name}/test-coronal.csv', 
                                 header=None, names=['id', 'coronal'])
        sagittal_df = pd.read_csv(f'./preds/{self.exp_name}/test-sagittal.csv', 
                                  header=None, names=['id', 'sagittal'])
        label_df = pd.read_csv(os.path.join(self.data_root, f'test-{self.task}.csv'), 
                               header=None, names=['id', 'label'])

        merged_df = axial_df.merge(coronal_df, on='id').merge(sagittal_df, on='id').merge(label_df, on='id')
        X = merged_df[['axial', 'coronal', 'sagittal']]
        y = merged_df['label']
        
        y_preds = self.logistic_regression.predict(X)
        
        metric = Metric()
        metric(torch.tensor(y), torch.tensor(y_preds))
        metric_result = metric.update()
        for key, value in metric_result.items():
            print(f"{key}: {value}")

    
    def save_model(self):
        with open(f'./models/lr_{EXP_NAME}_{TASK}.pkl', 'wb') as file:
            pickle.dump(self.logistic_regression, file)

if __name__ == '__main__':
    with open('fusion_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    DATA_ROOT = config['DATA_ROOT']
    TASK = config['TASK']
    MODEL_PATH = config['MODEL_PATH']
    EXP_NAME =  config['EXP_NAME']
    FOLD_NUM = config['FOLD_NUM']
    
    fusion_model = FusionModel(DATA_ROOT, TASK, MODEL_PATH, FOLD_NUM, EXP_NAME)
    fusion_model.build_dataset()
    fusion_model.fit()
    fusion_model.evaluate()
    fusion_model.save_model()
