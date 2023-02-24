from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import os
import time
import gc
import einops as ein
import torch.nn.functional as F
from utils.metric import micro_f1
import wandb

class BaselineTrainer():
    """
    훈련과정입니다.
    """
    def __init__(self, model, optimizer, device, save_dir,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None, epochs=1, tokenizer=None):
        self.model = model
        self.criterion = F.cross_entropy
        self.metric = micro_f1
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.tokenizer = tokenizer

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
        torch.cuda.empty_cache()
        del self.model, self.train_dataloader, self.valid_dataloader
        gc.collect()
    
    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        epoch_loss = 0
        steps = 0
        pbar = tqdm(self.train_dataloader)
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            steps += 1
            logits = self.model(input_ids = batch["input_ids"].to(self.device),
                                attention_mask = batch["attention_mask"].to(self.device))
            label = batch["labels"].squeeze().to(self.device)
            value = F.softmax(logits)
            loss = self.criterion(value, label)    
            loss.backward()

            epoch_loss += loss.detach().cpu().numpy().item()
            self.optimizer.step()
            wandb.log({'train_loss' : epoch_loss / steps})
            pbar.set_postfix({
                'loss' : epoch_loss / steps,
                'lr' : self.optimizer.param_groups[0]['lr'],
            })
        pbar.close()

    def _valid_epoch(self, epoch):
        val_loss = 0
        val_steps = 0
        val_score = 0 
        self.model.eval()
        val_loss_values= 2
        with torch.no_grad():
            
            for valid_batch in tqdm(self.valid_dataloader):
                val_steps += 1
                logits = self.model(valid_batch["input_ids"].to(self.device),
                                    valid_batch["attention_mask"].to(self.device))
                label = valid_batch["labels"].squeeze().to(self.device)
                value = F.softmax(logits)
                loss = self.criterion(value, label)    

                val_loss += loss.detach().cpu().numpy().item()
                label = label.detach().cpu().numpy()
                
                score = micro_f1(value.argmax(dim=1).detach().cpu().numpy(), label)
                val_score += score

            val_loss /= val_steps
            val_score /= val_steps 
            wandb.log({'val_loss':val_loss})
            wandb.log({'val_acc':val_score})
            print(f"Epoch [{epoch+1}/{self.epochs}] Val_loss : {val_loss}")
            print(f"Epoch [{epoch+1}/{self.epochs}] Val_score : {val_score}")

            if val_loss_values >= val_loss:
                print('save checkpoint!')
                if not os.path.exists(f'save/{self.save_dir}'):
                    os.makedirs(f'save/{self.save_dir}')
                torch.save(self.model.state_dict(), f'save/{self.save_dir}/model.pt')
                val_loss_values = val_loss