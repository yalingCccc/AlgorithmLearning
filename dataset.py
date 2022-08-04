import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import *
from sklearn.decomposition import PCA
from tqdm import tqdm
from gensim.models import Word2Vec
import os


def input_process(df, verbose=True):
    for feat in SEQFEATURES + MASKFEATURES:
        if verbose:
            print("process %s..." % feat)
        df[feat] = df[feat].apply(lambda x: np.fromstring(x, dtype=int, sep='|'))
    return df[FEATURES].values, df[ACTIONS].values


class MyDataset(Dataset):
    def __init__(self, X, y):
        super(MyDataset, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        out_x = []
        for x_i in x:
            if isinstance(x_i, np.ndarray):
                out_x.append(torch.from_numpy(x_i).to(device))
            else:
                out_x.append(torch.tensor(x_i).to(device))
        out_y = []
        for index, act in enumerate(ACTIONS):
            out_y.append(torch.tensor(y[index]).to(device))
        return out_x, out_y


class DataReader(object):
    def __init__(self, data_path, batch_size, shuffle=False, nrows=None, chunksize=None, verbose=False):
        self.data_path = data_path
        self.nrows = nrows
        self.chunksize = chunksize
        self.verbose = verbose
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader_num = None

    def get_dataloader(self, df):
        X, y = input_process(df, self.verbose)
        dataset_ = MyDataset(X, y)
        dataloader_ = DataLoader(dataset_, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader_

    def read(self):
        if self.chunksize is None:
            if self.nrows is None:
                data = pd.read_csv(self.data_path)
            else:
                data = pd.read_csv(self.data_path, nrows=self.nrows)
            self.len = data.shape[0]
            self.feature_num = data.shape[1]
            self.batch_num = math.ceil(self.len / self.batch_size)
            dataloader_ = self.get_dataloader(data)
            yield dataloader_
        else:
            if self.nrows is None:
                data = pd.read_csv(self.data_path, chunksize=self.chunksize)
            else:
                data = pd.read_csv(self.data_path, chunksize=self.chunksize, nrows=self.nrows)
            self.len = self.nrows
            for d in data:
                self.feature_num = d.shape[1]
                dataloader_ = self.get_dataloader(d)
                yield dataloader_

if __name__ == '__main__':
    val_dataset = DataReader(args.test_dataset, batch_size=args.batch_size, shuffle=False, nrows=args.val_size)
    dataloaders = val_dataset.read()
    for loader_i in dataloaders:
        for x,y in loader_i:
            print(x, y)