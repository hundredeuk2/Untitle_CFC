import torch
from torch import nn
from transformers import AutoModel

class BaseModel(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes=43,
                 dr_rate=0.2,):

        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dr_rate = dr_rate
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dr_rate),
            nn.Linear(self.model.config.hidden_size, num_classes)
        )

    def forward(self,  input_ids, attention_mask):
        _, pooler = self.model(input_ids = input_ids, attention_mask = attention_mask).to_tuple()
        logits = self.classifier(pooler)

        return logits

class CustomModel(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes=43,
                 dr_rate=0.2,):

        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dr_rate = dr_rate
        self.gru = nn.GRU(input_size=self.model.config.hidden_size,
                          hidden_size=self.model.config.hidden_size,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)
        self.activation = nn.Tanh()
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dr_rate),
            nn.Linear(self.model.config.hidden_size*2, num_classes)
        )

    def forward(self,  input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        _, hidden = self.gru(outputs) 
        outputs = torch.cat([hidden[-1], hidden[-2]], dim=1)
        outputs= self.activation(outputs)
        logits = self.regressor(outputs)
        return logits   