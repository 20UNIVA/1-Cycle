import json
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForQuestionAnswering, PreTrainedTokenizerFast
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
        
class QA_data():
    def __init__(self, max_len, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __call__(self, instance):
        context, question, start_idx, end_idx = instance
        
        inputs = self.tokenizer(context, question,
                                return_tensors="pt",
                                truncation=True,
                                max_length=self.max_seq_len,
                                padding='max_length',
                                add_special_tokens=True)
        target = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        start = torch.tensor(start_idx)
        end = torch.tensor(end_idx)

        return target.squeeze(), attention_mask.squeeze(), start, end

        
class QA_dataloader():
    def __init__(self, data, batch_size, tokenizer, max_len):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_processing = albert_data(self.max_len, self.tokenizer)

        
    def __iter__(self): # iterator to load data
        idx = 0
        
        #while(idx < len(self.data)):
        while True:
            batch = []
            for i in range(self.batch_size):
                if idx < len(self.data):
                    jdx = 0
                    if jdx < len(self.data[idx]['labelingData']):
                        context = self.data[idx]['refinedData']
                        question = self.data[idx]['labelingData'][jdx]['annotation'][0]['value']
                        start = self.data[idx]['labelingData'][jdx]['answer_start_index']
                        end = self.data[idx]['labelingData'][jdx]['answer_end_index']
                        # SOP, sentence-order prediction
                        instance = (context, question, start, end)
                        result = self.data_processing(instance)
                    batch.append(result)
                    jdx+=1
                idx += 1

                # To Tensor
                batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
                yield batch_tensors        
        
        
class QA_dataset():
    def __init__(self, tokenizer, data, max_seq_len = 512):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        inputs = self.tokenizer(self.data[idx]['context'], self.data[idx]['question'],
                                return_tensors="pt",
                                truncation=True,
                                max_length=self.max_seq_len,
                                padding='max_length',
                                add_special_tokens=True)
        target = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        start = torch.tensor(self.data[idx]['start'])
        end = torch.tensor(self.data[idx]['end'])

        return target.squeeze(), attention_mask.squeeze(), start, end
            
    
    
class QA(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = BartForQuestionAnswering.from_pretrained('hyunwoongko/kobart')
        
        self.lr = args.learning_rate
        
    def forward(self, batch):
        target, attention_mask, start, end = train_batch
        outputs = self.model(input_ids = target, attention_mask = attention_mask, start_positions=start, end_positions=end)
        return outputs.logits
        
    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) #0.00002 로 바꿔보기
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        target, attention_mask, start, end = train_batch
        #outputs = self.model(input_ids = x, attention_mask = y, labels = z)
        outputs = self.model(input_ids = target, attention_mask = attention_mask, start_positions=start, end_positions=end)
        loss = outputs.loss

        self.log('train_loss', loss, on_epoch = True)
            
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        target, attention_mask, start, end = val_batch
        #outputs = self.model(input_ids = x, attention_mask = y, labels = z)
        outputs = self.model(input_ids = target, attention_mask = attention_mask, start_positions=start, end_positions=end)
        # pred = torch.argmax(outputs, dim = 1)
        # orign = torch.argmax(z, dim = 1)
        # return pred == orign
        
#     def validation_epoch_end(self, validation_step_outputs):
#         answers=[]
#         for out in validation_step_outputs:
#             for i in out:
#                 answers.append(i)
#         accuracy = sum(answers)/len(answers)
        loss = outputs.loss
        self.log('val_loss', loss)
    
def main():
    #### LM pretrining 다시 시킨 뒤 새로 코드 업뎃 예정
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_name', type=str, default='test', help='')
    parser.add_argument('--root_save_dir', type=str, default='./QA/BartQA', help='')
    parser.add_argument('--data_dir', type=str, default='tmp/data_QA_1046_revised', help='')                                    
    
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of text')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument("--learning_rate", default=3.5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=int, dest="max_epochs", default=100)
    
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'  # Set the GPUs 0 and 1 to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    with open(args.data_dir+'.json', 'r') as f:
        QA_json = json.load(f)

    QA_datas = []
    for i in range(len(QA_json)):
        for j in range(len(QA_json[i]['labelingData'])):
            context = QA_json[i]['refinedData']
            quest = QA_json[i]['labelingData'][j]['annotation'][0]['value']
            answer = QA_json[i]['labelingData'][j]['annotation'][1]['value']
            ans_idx = context.find(answer)
            
            QA_datas.append({'context':context,'question':quest,
                             'start':ans_idx,'end':ans_idx+len(answer)})
            
            
    val_ratio = 0.95
    train_len = int(len(QA_datas)*val_ratio)
    QA_train = QA_datas[:train_len]
    QA_valid = QA_datas[train_len:]
    tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
    QA_train_dataset = QA_dataset(tokenizer, QA_train)
    QA_train_dataloader = DataLoader(QA_train_dataset, batch_size=64, shuffle=True)
    QA_valid_dataset = QA_dataset(tokenizer, QA_valid)
    QA_valid_dataloader = DataLoader(QA_valid_dataset, batch_size=1, shuffle=False)
    
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='checkpoints_{epoch:d}',
        verbose=True,
        save_last=True,
        save_top_k=3,
        monitor='val_loss',
        mode='min',
    )

    QA_model = QA(args)
    trainer = pl.Trainer(precision=16, accelerator='gpu', devices =2, max_epochs = args.max_epochs, strategy='ddp',  accumulate_grad_batches= 64, default_root_dir=args.root_save_dir, callbacks=[checkpoint_callback])
    trainer.fit(QA_model, QA_train_dataloader, QA_valid_dataloader)


if __name__ == '__main__':
    main()