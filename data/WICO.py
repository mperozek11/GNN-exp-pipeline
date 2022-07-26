import torch
from torch_geometric.data import InMemoryDataset, download_url
from tensorflow import keras
import pathlib
import sys

root = pathlib.Path().resolve().as_posix()
sys.path.insert(0, f'{root}/GNN-exp-pipeline/transforms')
from wico_transforms import MultiTargetData


class WICO(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{root}/GNN-exp-pipeline/data/full_wico.pt']

    @property
    def processed_file_names(self):
        return ['processed_wico.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass
    def process(self):
        # Read data into huge `Data` list.
        data_list = torch.load(self.raw_file_names[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# ============================================================
# ======================== FILTERS ===========================
# ============================================================

def filter_5g_non(data):
    if data.y == 2:
        return False
    return True


# ============================================================
# ======================= TRANSFORMS =========================
# ============================================================

def wico_data_to_custom(data):
    y = torch.tensor(keras.utils.to_categorical(data.y, 2))
    return MultiTargetData(x=data.x, edge_index=data.edge_index, y=y)

    