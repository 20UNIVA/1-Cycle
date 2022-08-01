import json
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartModel, PreTrainedTokenizerFast, BartForConditionalGeneration

import pytorch_lightning as pl
import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
        
class summary_dataset(Dataset):
    def __init__(self, tokenizer, data, max_seq_len = 512):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.data[idx]['refinedData'],
                                return_tensors="pt",
                                truncation=True,
                                max_length=self.max_seq_len,
                                padding='max_length',
                                add_special_tokens=True)
        summ = self.data[idx]['labelingData'][0]['annotation'][0]['value']
        
        target = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        label = self.tokenizer(summ,
                            return_tensors="pt",
                            truncation=True,
                            max_length=128,
                            padding='max_length',
                            add_special_tokens=True)["input_ids"]
        
        
        return target.squeeze(), attention_mask.squeeze(), label.squeeze()


    
class summary(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart')
        
        self.lr = args.learning_rate
        
    def forward(self, batch):
        inputs, label = batch
        outputs = self.model(**inputs, label=label)
        return outputs.logits
        
    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) #0.00002 로 바꿔보기
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y, z = train_batch
        outputs = self.model(input_ids = x, attention_mask = y, labels = z)
        loss = outputs.loss

        self.log('train_loss', loss, on_epoch = True)
            
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y, z = val_batch
        outputs = self.model(input_ids = x, attention_mask = y, labels = z)
        loss = outputs.loss

        self.log('val_loss', loss, on_epoch = True)
            
        return loss
    
def main():
    #### LM pretrining 다시 시킨 뒤 새로 코드 업뎃 예정
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_name', type=str, default='test', help='')
    parser.add_argument('--root_save_dir', type=str, default='./summary', help='')
    parser.add_argument('--data_dir', type=str, default='tmp/data_summary_345_revised.json', help='')
    parser.add_argument('--pretrained_model', type=str, default='../LM_pretraining/model_albert/lightning_logs/version_4/checkpoints/checkpoints_epoch=0_step=5087.ckpt', help='')
                                              
    
    parser.add_argument('--max_length', type=int, default=512, help='Maximum length of text')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument("--learning_rate", default=3.5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=int, dest="max_epochs", default=100)
    #parser.add_argument("--pretrained", type=str2bool, default=False)
    
    #parser.add_argument("--num_device", type=int, default = 1)
    #parser.add_argument("--device", type=str)
    
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'  # Set the GPUs 0 and 1 to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    with open(args.data_dir, 'r') as f:
        summary_datas = json.load(f)

    val_ratio = 0.8
    train_len = int(len(summary_datas)*val_ratio)
    summary_train = summary_datas[:train_len]
    summary_valid = summary_datas[train_len:]
    tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
    summary_train_dataset = summary_dataset(tokenizer, summary_train)
    summary_train_dataloader = DataLoader(summary_train_dataset, batch_size=64, shuffle=True)
    summary_valid_dataset = summary_dataset(tokenizer, summary_valid)
    summary_valid_dataloader = DataLoader(summary_valid_dataset, batch_size=1, shuffle=False)
    #kobert_tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='checkpoints_{epoch:d}',
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )

    summary_model = summary(args)
    trainer = pl.Trainer(precision=16, accelerator='gpu', devices =2, max_epochs = args.max_epochs, strategy='ddp',  accumulate_grad_batches= 64, default_root_dir=args.root_save_dir, callbacks=[checkpoint_callback])
    trainer.fit(summary_model, summary_train_dataloader, summary_valid_dataloader)


if __name__ == '__main__':
    main()