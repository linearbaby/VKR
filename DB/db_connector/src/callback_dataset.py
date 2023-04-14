from torch.utils.data import Dataset


class CallbackDataset(Dataset):
    def __init__(self, getitem_callback, len):
        super().__init__()
        self.len = len
        self.getitem_callback = getitem_callback

    def __getitem__(self, idx):
        return self.getitem_callback(idx)

    def __len__(self):
        return self.len
