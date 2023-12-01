
from tqdm import tqdm
import os
import pandas as pd


#shuffling is on
if __name__ == '__main__':
    train_ratio=0.8
    valid_ratio=0.1
    dataset = pd.read_csv("/home/yerinyoon/Cubigate_ai_engine/dp/data/adult.csv")
    print(dataset.shape)
    dataset_len=dataset.shape[0]
    train_len=int(dataset_len*train_ratio)
    valid_len=int(dataset_len*valid_ratio)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    train_data=dataset.iloc[:train_len].astype(str)
    valid_data=dataset.iloc[train_len:train_len+valid_len].astype(str)
    test_data=dataset.iloc[train_len+valid_len:].astype(str)


    train_data.to_csv('adult_train.csv', index=False)
    test_data.to_csv('adult_test.csv', index=False)
    valid_data.to_csv('adult_val.csv', index=False)
