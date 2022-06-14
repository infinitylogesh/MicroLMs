
#%%
import numpy as np
import gzip
import torch


def read_enwik8(path=None, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):

    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.fromstring(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

trX,vaX,teX = read_enwik8(path="/Users/logesh.umapathi/garage/personal/MicroLMs/MicroLMs/data/enwik8.gz")
trX.size()

#%%
def sample_batch(data,batch_size,seq_length):
    starts = torch.randint(size=(batch_size,),low=0,high=data.size()[0]-seq_length-1)
    ends = starts+seq_length
    seq_inputs = [data[start:end] for start,end in zip(starts,ends)]
    seq_targets = [data[start+1:end+1] for start,end in zip(starts,ends)]
    inputs = torch.stack(seq_inputs)
    targets = torch.stack(seq_targets)
    return inputs,targets


#sample_batch(trX,batch_size=32,seq_length=128)