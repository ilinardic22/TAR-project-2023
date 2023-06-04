import numpy as np
import torch

def embedding_matrix(vocab, seed, embeddings_path=None, freeze=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    embeddings=np.random.normal(0,1,(len(vocab), 300))

    embeddings[0]=np.zeros((1,300))
    embeddings[1]=np.ones((1,300))

    if embeddings_path!=None:

        file=open(embeddings_path, "r")

        lines={}

        for line in file.readlines():

            line=line.strip("\n").split()

            lines[line[0]]=[float(i) for i in line[1:]]
        
        keys=lines.keys()

        words_in_vocab=list(vocab.stoi.keys())

        for i in range(2, len(words_in_vocab)):

            if words_in_vocab[i] in keys:

                embeddings[i]=lines[words_in_vocab[i]]
            

    embeddings=torch.nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float), padding_idx=0, freeze=freeze)

    return embeddings

