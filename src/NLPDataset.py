from torch.utils.data import Dataset
from Vocab import Vocab

class NLPDataset(Dataset):

    def __init__(self, word_vocab:Vocab, label_vocab:Vocab, path):
        super().__init__()
        self.word_vocab=word_vocab
        self.label_vocab=label_vocab
        self.instances=[]

        file=open(path, "r")

        for line in file.readlines():

            line=line.strip("\n").split(",")

            text=line[0].split()
            label=line[1].strip()

            self.instances.append([text, label])


    def __getitem__(self, index):

        instance_text, instance_label=self.instances[index]

        return (self.word_vocab.encode(instance_text), self.label_vocab.encode(instance_label))

    def __len__(self):
        return len(self.instances)


