import torch

class Vocab:

    def __init__(self, frequencies, max_size, min_freq, add_special_tokens=False):
        current_size=0

        self.stoi={}
        self.itos={}

        if add_special_tokens:

            self.stoi={"<PAD>":0,
                    "<UNK>":1}
            self.itos={0:"<PAD>",
                    1:"<UNK>"}
            current_size=2
        

        freq=[(key, value) for key, value in frequencies.items() if value>=min_freq]

        sorted_freqs=sorted(freq, key=lambda x: x[1], reverse=True)

        if max_size==-1:

            for word, _ in sorted_freqs:

                self.stoi[word]=current_size
                self.itos[current_size]=word

                current_size+=1
        
        else:
            
            for word, _ in sorted_freqs:
                if current_size==max_size:
                    break
                else:
                    self.stoi[word]=current_size
                    self.itos[current_size]=word

                    current_size+=1                

    def encode(self, text):

        if isinstance(text, list):

            enc_text=[self.stoi.get(i, 1) for i in text]

            return torch.tensor(enc_text)

        return torch.tensor(self.stoi.get(text, 1))

    def __len__(self):
        return len(self.stoi.keys())


