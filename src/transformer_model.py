import torch.nn as nn
from transformers import BertModel

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


class PersonalityClassifier(nn.Module):
    def __init__(self, n_classes):
        super(PersonalityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        output = self.drop(pooled_output)
        return self.out(output)
