import torch
from torch.utils.data import Dataset, DataLoader

import logging
logger = logging.getLogger(__name__)

def load_data(path):
    '''
    Function for load data from path
    '''
    data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                data.append(str(line))
    except Exception:
        logger.exception("Failed to load file: %s", path)
        raise

    return data

class JokesDataset(Dataset):
    '''
    Dataset class for getting token sequence
    '''
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        line = self.texts[idx]
        tokens = self.tokenizer.encode(line)
        return torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch, pad_id=0, model_max_length=128):
    '''
    Collate function in dataloader for process sequences to tensor batch
    '''
    truncated = []
    for line in batch:
        truncated.append(line[:model_max_length])
    
    max_length = max(len(line) for line in truncated)

    input_ids = torch.full(
        (len(truncated), max_length),
        pad_id,
        dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)

    for i, line in enumerate(truncated):
        length = len(line)

        input_ids[i, :length] = line
        attention_mask[i, :length] = 1
    
    return input_ids, attention_mask