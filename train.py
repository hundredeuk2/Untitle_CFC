from utils import seed, wandb
import torch
import torch.optim as optim
import argparse
from dataloader.custom_dataloader import BERTDataset
from model.custom_model import CustomModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from trainer.base_trainer import BaselineTrainer
import pandas as pd
from sklearn.model_selection import train_test_split


def main(config):
    seed.seed_everything(config.train.seed)
    df = pd.read_excel(config.data.data_path)
    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    # 데이터셋 로드 클래스를 불러옵니다.
    train = BERTDataset(train, config.model.model_name)  
    valid = BERTDataset(test, config.model.model_name, 'test')
    
    train_dataloader = DataLoader(train, batch_size= config.train.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid, batch_size= config.train.batch_size, shuffle=False)
    
    # GPU 사용 설정을 해줍니다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 아키텍처를 불러옵니다.
    print(f'현재 적용되고 있는 모델은 {config.model.model_class}입니다.')
    model = CustomModel(config.model.model_name).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    
    lr_scheduler = None
    epochs = config.train.max_epoch
    
    trainer = BaselineTrainer(
            model = model,
            optimizer = optimizer,
            device = device,
            train_dataloader = train_dataloader,
            valid_dataloader = valid_dataloader,
            epochs=epochs,
            save_dir= 'Baseline'
        )
    
    trainer.train()

if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()

    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')
    config_w = wandb.wandb_setting(entity = "hundredeuk2",
                             project = "Feedback-classifier",
                             group_name = "roberta-large",
                             experiment_name = "Baseline_model",
                             arg_config = args.config)
    main(config_w)