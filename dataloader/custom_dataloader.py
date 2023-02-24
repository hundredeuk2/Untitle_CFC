import torch
import numpy as np
import pandas as pd
import einops as ein
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class BERTDataset(Dataset):
    def __init__(self, path, model_name, mode='train', max_len=512):
        # self.df = df
        df = pd.read_excel(path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mode = mode
        self.max_length = max_len
        self.encoder = LabelEncoder()
        self.label = df['최종분류(우선순위 가장 높은것 선택)']
        self.encoder.fit(self.label)
        train, valid = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
        train.reset_index(inplace=True, drop=True)
        valid.reset_index(inplace=True, drop=True)

        if self.mode == "train":
            self.sentence, self.label = self.load_data(train)
        else:
            self.sentence, self.label = self.load_data(valid)

    def load_data(self, data:pd.DataFrame):
        sentence = data['발화'].to_numpy()
        label = self.encoder.transform(self.label)
        return sentence, label

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

        return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s'), 
                    'labels': ein.rearrange(torch.tensor(self.label[i], dtype=torch.long), ' -> 1')}
    

    def __len__(self):
        return (len(self.sentence))