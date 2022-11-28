#import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch, os, collections
import pandas as pd
import pickle5 as pickle
from tqdm import tqdm


class ProtEmbDataset(Dataset):
    """Protein emb dataset."""
    def __init__(self, model_name, transform=None):
        """
        Args:
            root_dir (string): Directory with all the datapoints.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = '/media/dell4/a87d4a0b-5641-40d3-8d10-416948fc2996/ION_DATA/%s/'%(model_name)
        self.filenames = os.listdir(self.data_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_ = self.filenames[idx]
        with open(self.data_path+file_, 'rb') as handle:
            datapoint = pickle.load(handle)
        embs = datapoint['embs']
        labels = datapoint['labels']
        return embs, labels