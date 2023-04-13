import torch
import numpy as np
import pandas as pd
import einops as ein
from typing import Callable

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, 
                df: str,
                tokenizer : Callable,
                max_length : int = 128
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.text = self.load_data(df)

    def load_data(self, df):
        text = df['sentence_1'].to_numpy()
        return text

    def __getitem__(self, idx):
        sentence_1 = self.text
        
        encoded_dict = self.tokenizer.encode_plus(
            sentence_1[idx],
            add_special_tokens = True,      
            max_length = self.max_length,           
            padding='max_length',
            truncation=True,
            return_attention_mask = True,   
            return_tensors = 'pt',          
            )
        
        return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s')}
        
    def __len__(self):
        return len(self.text)