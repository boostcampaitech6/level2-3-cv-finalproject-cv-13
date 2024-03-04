import json
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import csv
from tqdm import tqdm
import shutil

classes = {0 : [0,0,0], 1 : [0,0,1], 2 : [0,1,0], 3 : [0,1,1], 4 : [1,0,0], 5 : [1,0,1], 6 : [1,1,0], 7 : [1,1,1]}
# [abnormal, acl, meniscus]

def get_df(task, fold_num, train=True):
    abnormal_file = "./data/train-abnormal.csv"
    acl_file = "./data/train-acl.csv"
    meniscus_file = "./data/train-meniscus.csv"

    abnormal_df = pd.read_csv(abnormal_file, names=['id', 'abnormal'])
    acl_df = pd.read_csv(acl_file, names=['id', 'acl'])
    meniscus_df = pd.read_csv(meniscus_file, names=['id', 'meniscus'])

    df = pd.concat([abnormal_df, acl_df['acl'],meniscus_df['meniscus']], axis=1)

    df['class'] = [[k for k, v in classes.items() 
                        if v == [df['abnormal'].iloc[i], df['acl'].iloc[i], df['meniscus'].iloc[i]]][0]
                        for i in range(df.shape[0])]
    
    all = df.to_numpy()
    X = np.ones(all.shape[0])
    y = np.array([v[-1] for v in all])
    groups = np.array([v[0] for v in all])

    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=2024)

    train_fold = []
    valid_fold = []

    for train_idx, val_idx in cv.split(X, y, groups):
        train_fold.append(groups[train_idx])
        valid_fold.append(groups[val_idx])

    train_id = train_fold[fold_num]
    valid_id = valid_fold[fold_num]
    

    if train:
        train_line = [line for line in all if line[0] in train_id]
        train_csv = pd.DataFrame(train_line, columns=df.columns)
        train_csv = train_csv[['id', task]]
        train_csv.rename(columns={task:'label'}, inplace=True)
        print(task, train)
        return train_csv
    else:
        valid_line = [line for line in all if line[0] in valid_id]
        valid_csv = pd.DataFrame(valid_line, columns=df.columns)
        valid_csv = valid_csv[['id', task]]
        valid_csv.rename(columns={task:'label'}, inplace=True)
        print(task, train)
        return valid_csv
    


    
    