from torch_geometric.loader import DataLoader
from torch.utils.data import  random_split
from dataclasses import dataclass
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
import torch


def build(
    dataset,
    batch_size=64,
    train_size=0.8,
    valid_size=0.1,
    collate_fn = None,
):
    return Builder(
        dataset=dataset,
        batch_size=batch_size,
        train_size=train_size,
        valid_size=valid_size,
        collate_fn = collate_fn,
    )

@dataclass
class Builder:
    train: DataLoader
    valid: DataLoader
    test: DataLoader

    def __init__(
            self,
            dataset : list,
            batch_size : int = 64,
            train_size : float = 0.8,
            valid_size : float = 0.1,
            collate_fn = None,
    ):
        n = len(dataset)
        print (f"\nDataset size: {n}\n")

        train_len = int(train_size * n)
        valid_len = int(valid_size * n)
        test_len = n - train_len - valid_len

        train_ds, valid_ds, test_ds = random_split(
            dataset,
            [train_len, valid_len, test_len]
        )
    
        self.train = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        self.valid = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        self.test = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
 
    def collate_fn(batch):
        '''    
        PyG style
        '''   
        batch = Batch.from_data_list(batch)

        # y stacking (safe)
        if hasattr(batch, "y"):
            batch.y = torch.stack(batch.y) if isinstance(batch.y, list) else batch.y

        # task_id stacking
        if hasattr(batch, "task_id"):
            batch.task_id = torch.stack([
                t if torch.is_tensor(t) else torch.tensor(t)
                for t in batch.task_id
            ])

        return batch

