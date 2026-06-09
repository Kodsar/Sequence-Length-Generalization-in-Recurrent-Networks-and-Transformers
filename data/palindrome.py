import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

VOCAB_SIZE = 10
PAD_TOKEN = 10
DEV_SIZE = 30_000 
TEST_SIZE = 10_000

DEV_SEQ_RANGE = (50, 100)
TEST_SEQ_RANGE = (300, 500) 

class PalindromeDataset(Dataset):
  
    def __init__(self, size: int, seq_range: tuple, seed: int):

        self.min_len, self.max_len = seq_range
        self.rng = np.random.default_rng(seed)
        self.data = []
        
        n_pos = size // 2
        n_neg = size - n_pos
        
        # --- Generate Positives ---
        for _ in range(n_pos):
            length = self.rng.integers(self.min_len, self.max_len + 1)
            is_odd = (length % 2 == 1)
            half_len = length // 2
            half_seq = self.rng.integers(0, VOCAB_SIZE, size=half_len)
            
            if is_odd:
                mid_char = self.rng.integers(0, VOCAB_SIZE, size=1)
                seq = np.concatenate([half_seq, mid_char, half_seq[::-1]])
            else:
                seq = np.concatenate([half_seq, half_seq[::-1]])
            
            self.data.append((torch.from_numpy(seq).long(), torch.tensor(1, dtype=torch.long)))

        # --- Generate Negatives ---
        count = 0
        while count < n_neg:
            length = self.rng.integers(self.min_len, self.max_len + 1)
            cand = self.rng.integers(0, VOCAB_SIZE, size=length)
            if np.array_equal(cand, cand[::-1]): continue
            self.data.append((torch.from_numpy(cand).long(), torch.tensor(0, dtype=torch.long)))
            count += 1
            
        self.rng.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]

def collate_fn(batch):

    inputs, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs])
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    targets = torch.stack(targets)
    return inputs_padded, targets, lengths

def get_base_datasets(seed=100):

    dev_ds = PalindromeDataset(DEV_SIZE, DEV_SEQ_RANGE, seed=seed)
    test_ds = PalindromeDataset(TEST_SIZE, TEST_SEQ_RANGE, seed=seed+100)
    return dev_ds, test_ds