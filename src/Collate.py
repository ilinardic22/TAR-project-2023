from torch.nn.utils.rnn import pad_sequence
import torch

def pad_collate_fn(batch, pad_index=0):

    texts, labels=zip(*batch)

    padded=pad_sequence(texts, batch_first=True, padding_value=0)

    lengths=torch.tensor([i.size(dim=0) for i in texts])

    return (padded, torch.tensor(labels), lengths)

        

    