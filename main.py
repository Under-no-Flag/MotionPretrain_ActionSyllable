import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.h36m import Human36mDataset
from model.MotionGPT import MotionGPT,MotionLoss
import numpy as np
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='Motion GPT.')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='../data/h3.6m', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='checkpoints/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser
    return parser

class Processor:

    def __init__(self,args):
        self.args=args

        self.args = args
        self.device = torch.device(args.device)

        self.load_model()
        self.load_data()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',     # 监控验证损失的最小化
            patience=3,     # 3个epoch无改善后调整学习率
            factor=0.5,    # 学习率衰减比例为0.5
            verbose=True    # 打印调整信息
        )
        self.criterion = MotionLoss()

    def load_model(self):
        self.model = MotionGPT(
            d_model=64,
            n_heads=4,
            num_layers=10,
            max_seq_len=50,
            input_dim=6,
            output_dim=6
        ).to(self.device)


    def load_data(self):
        self.train_set = Human36mDataset(
            data_dir=self.args.data_dir,
            split='train',
            input_length=50,
            predicted_length=25
        )
        self.val_set = Human36mDataset(
            data_dir=self.args.data_dir,
            split='val',
            input_length=50,
            predicted_length=25
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, targets, _,_ in tqdm(self.train_loader, desc='Training'):
            inputs = inputs.float().to(self.device)
            targets = targets.float().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets, _,_ in tqdm(self.val_loader, desc='Validating'):
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def start(self):

        best_val_loss = float('inf')
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # 根据验证损失更新学习率
            self.scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}/{self.args.epochs}')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            #print lr
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'{self.args.save_dir}/best_model.pth')
                print('Saved best model')

def main():

    parser = get_parser()
    p=parser.parse_args()

    if p.config is not None:
        default_arg = yaml.load(open(p.config, 'r'), Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)


    args= parser.parse_args()
    import os
    os.makedirs(args.save_dir, exist_ok=True)

    processor=Processor(args)
    processor.start()
if __name__=="__main__":

    main()