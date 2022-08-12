import torch
from torch.utils.data import Dataset
import pathlib

root = pathlib.Path().resolve().as_posix()

class motifWICO(Dataset):

    def __init__(self, root_dir, pre_filter=None, pre_transform=None):
        self.RAW_DATA_PATH = f'{root}/GNN-exp-pipeline/data/motif_wico/motif_wico.pt'
        self.root_dir = root_dir
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.dataset = torch.load(self.RAW_DATA_PATH)
        if self.pre_filter:
            self.dataset = [item for item in self.dataset if self.pre_filter(item)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.dataset[idx, :-1]
        labels = self.dataset[idx, -1]
        sample = {'x': features, 'y': labels}
        
        if self.pre_transform:
            sample = self.transform(sample)

        return sample