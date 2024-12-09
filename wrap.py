from torch.utils.data import Dataset
import torch

class WrapDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Flatten the list of tuples
        self.flat_data = [item for sublist in self.data for item in sublist]

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, index):
        x, t = self.flat_data[index]
        return torch.tensor(x, dtype=torch.long), torch.tensor(t, dtype=torch.long)
