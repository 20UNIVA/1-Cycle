import json
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer #LongformerConfig, LongformerForQuestionAnswering, PreTrainedTokenizerFast
import pytorch_lightning as pl
import argparse
import os
from collections import Counter

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')     

def QA_F1(truth, pred):
    
    #truth O, pred O
    tp = sum((Counter(truth) & Counter(pred)).values())
    #truth X, pred O
    fp = sum((Counter(pred) - Counter(truth)).values())
    #truth O, pred X
    fn = sum((Counter(truth) - Counter(pred)).values())

    #정답이 없는 경우, 둘이 일치하면 1 아니면 0 반환
    if (len(pred) == 0) or (len(truth) == 0):
        return int(pred == truth)
    elif tp == 0:
        return 0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall)/(precision + recall)
    
    return f1

def EM(truth, pred):
    if (len(pred) == 0) and (len(truth) == 0):
        return 1
    elif (len(pred) == 0) or (len(truth) == 0):
        return 0
    else:
        # print(pred == truth)
        return int(pred == truth)
    
    
    
        
class QA_dataset():
    def __init__(self, data, tokenizer, max_seq_len = 1024):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer     
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        context = self.data[idx]['context']
        inputs = self.tokenizer(context, self.data[idx]['question'],
                                return_tensors="pt",
                                truncation=True,
                                max_length=self.max_seq_len,
                                padding='max_length',
                                add_special_tokens=True)
        target = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        if self.data[idx]['start'] == -1:
            st_idx = 0
        else:
            st_idx = self.data[idx]['start']
        ed_idx = self.data[idx]['end']
        
        before = context[:st_idx]
        ans = context[st_idx:ed_idx]
        before_token = self.tokenizer.tokenize(before)
        ans_token = self.tokenizer.tokenize(ans)
        con_token = self.tokenizer.tokenize(context)
        
        ed_token_idx = len(before_token)+len(ans_token)
        st_token_idx = len(before_token)
        for i in range(len(ans_token)):
            if ans_token[i] == con_token[st_token_idx]:
                ed_token_idx = st_token_idx+len(ans_token[i:])
                break
        
        return target.squeeze(), attention_mask.squeeze(), torch.tensor(st_token_idx), torch.tensor(ed_token_idx), ans
            
    
    
class QA(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained('markussagen/xlm-roberta-longformer-base-4096')
        self.lr = 5e-5 
        #args.learning_rate
        
    def forward(self, batch):
        target, attention_mask, start, end = train_batch
        outputs = self.model(input_ids = target, attention_mask = attention_mask, start_positions=start, end_positions=end)
        return outputs.logits
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        target, attention_mask, start, end, _ = train_batch
        outputs = self.model(input_ids = target, attention_mask = attention_mask, start_positions=start, end_positions=end)
        loss = outputs.loss

        self.log('train_loss', loss, on_epoch = True)
            
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        target, attention_mask, start, end, truth = val_batch
        outputs = self.model(input_ids = target, attention_mask = attention_mask)
        pred_st = outputs.start_logits.argmax()
        pred_end = outputs.end_logits.argmax() + 1
        pred = target[0][pred_st:pred_end].tolist()
        # truth = target[0][start:end].tolist()
        f1 = QA_F1(truth, pred)
        self.log('val_f1', f1)
        em = EM(truth, pred)
        self.log('val_EM', em)
        loss = outputs.loss
        self.log('val_loss', loss)
        
        return loss
        
def filter_data(QA_json,tokenizer, max_length):
    new_json = []
    for data in tqdm(QA_json):
        context = data['context']
        inputs = tokenizer(context, data['question'],
                                return_tensors="pt",
                                add_special_tokens=True)
        target = inputs["input_ids"].squeeze()
        if data['start'] == -1:
            st_idx = 0
        else:
            st_idx = data['start']
        ed_idx = data['end']

        before = context[:st_idx]
        ans = context[st_idx:ed_idx]
        before_token = tokenizer.tokenize(before)
        ans_token = tokenizer.tokenize(ans)

        ed_token_idx = len(before_token)+len(ans_token)
        if ed_token_idx < max_length:
            new_json.append(data)
    return new_json
    
def main():
    #### LM pretrining 다시 시킨 뒤 새로 코드 업뎃 예정
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_name', type=str, default='test', help='')
    parser.add_argument('--root_save_dir', type=str, default='./QA/BartQA', help='')
    parser.add_argument('--data_path', type=str, default='./data_QA_1046_revised.json', help='')                                    
    parser.add_argument('--device', type=str, default='1')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of text')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument("--learning_rate", default=3.5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_epochs", type=int, dest="max_epochs", default=100)
    
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= args.device  # Set the GPUs 0 and 1 to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained('markussagen/xlm-roberta-longformer-base-4096', max_length = 4096, padding = 'max_length', truncation = True)
    
    with open(args.data_path, 'r') as f:
        QA_json = json.load(f)
                
    print('data#:',len(QA_json))
    val_ratio = 0.95
    train_len = int(len(QA_json)*val_ratio)
    QA_train = QA_json[:train_len]
    QA_valid = QA_json[train_len:]

    
    QA_train_dataset = QA_dataset(QA_train, tokenizer, args.max_length)
    QA_train_dataloader = DataLoader(QA_train_dataset, batch_size=args.batch_size, shuffle=True)
    QA_valid_dataset = QA_dataset(QA_valid, tokenizer, args.max_length)
    QA_valid_dataloader = DataLoader(QA_valid_dataset, batch_size=1, shuffle=False)
    
    QA_model = QA(args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='checkpoints_{epoch:d}',
        verbose=True,
        save_last=True,
        save_top_k=2,
        monitor='val_loss',
        mode='min',
    )

    
    trainer = pl.Trainer(precision=16, gpus=-1, max_epochs = args.max_epochs, accumulate_grad_batches= 64, default_root_dir=args.root_save_dir, callbacks=[checkpoint_callback])#, resume_from_checkpoint = './QA/Longformer_b16_lr5e-5/ligntning_logs/version_0/checkpoints/last.ckpt')
    trainer.fit(QA_model, QA_train_dataloader, QA_valid_dataloader)


if __name__ == '__main__':
    main()
    
    