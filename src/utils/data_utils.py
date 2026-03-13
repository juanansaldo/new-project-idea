from torch.utils.data import Dataset


class TransformWrapper(Dataset):
    """Wraps a dataset and applies a transform to the first element (image) in __getitem__."""


    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y