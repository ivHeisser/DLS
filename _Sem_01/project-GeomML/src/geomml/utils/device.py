import torch


def to_device(batch, device):
    '''
    move batch to device
    '''
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def get_device():
    '''
    determine and choice available devices
    '''
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")