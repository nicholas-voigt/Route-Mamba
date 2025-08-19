from torch.utils.data import Dataset
import torch
import pickle
import os

class TSP(object):    
    def __init__(self, size):
        self.size = size
        print(f'TSP with {self.size} nodes.')
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)


class TSPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0):
        super(TSPDataset, self).__init__()
        self.data = []
        self.size = size

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [self.make_instance(args) for args in data[offset:offset+num_samples]]
        else:
            self.data = [{'coordinates': torch.rand(self.size, 2)} for i in range(num_samples)]        
        self.N = len(self.data)
        print(f'{self.N} instances initialized.')
    
    def make_instance(self, args):
        return {'coordinates': torch.FloatTensor(args)}
    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]
