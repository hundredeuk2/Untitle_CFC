import torch
import numpy as np
import pandas as pd
import einops as ein
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class BERTDataset(Dataset):
    def __init__(self, df, model_name, mode='train', max_len=512):
        self.df = df
        # df = pd.read_excel(path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mode = mode
        self.max_length = max_len
        self.encoder = LabelEncoder()
        if self.mode == "train":
            self.sentence, self.label = self.load_data(self.df)
        else:
            self.sentence = self.load_data(self.df)

    def load_data(self, data:pd.DataFrame, train=True):
        sentence = data['발화'].to_numpy()
        if train:
            label = data['최종분류(우선순위 가장 높은것 선택)']
            self.encoder.fit(label)
            label = self.encoder.transform(label)
            return sentence, label
        return sentence

    def __getitem__(self, i):
        sentence = self.sentence

        encoded_dict = self.tokenizer.encode_plus(
            sentence[i],
            add_special_tokens = True,      
            max_length = self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask = True,   
            return_tensors = 'pt',
        )
        if self.mode == 'train':
            return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s'), 
                    'labels': ein.rearrange(torch.tensor(self.label[i], dtype=torch.long), ' -> 1')}
        else:
            return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s')}

    def __len__(self):
        return (len(self.sentence))