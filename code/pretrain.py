import json
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import  AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainingArguments, pipeline, AutoConfig
import sys, os


model = AutoModelForMaskedLM.from_pretrained(model_path) #MLM用モデル
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) #トークナイザーも学習する


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15 #マスキングする確率
)


dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = data_path,
    block_size = 512  # maximum sequence length
)

print('No. of lines: ', len(dataset)) # No of lines in your datset

training_args = TrainingArguments( #trainのargs
    output_dir 'output_dir_path,
    overwrite_output_dir=True,
    evaluation_strategy = 'no',
    gradient_accumulation_steps=1, #勾配累積のステップ数
    per_device_train_batch_size=32, #GPU一つに対するバッチサイズ
    
    save_strategy='epoch',
    
    num_train_epochs=10,
    
    fp16 = True, #混合精度学習のbool

    lr_scheduler_type='cosine', #大体cosineが最適な印象
    weight_decay=0.01,
    warmup_ratio=0.2,
)

trainer = Trainer( #trainerに渡す
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train() 


#trainerからモデルを保存 trainig-argsも保存
trainer.save_model(保存先path)

#モデルの保存
trainer.model.save_pretrained('mlm_model_path')

#トークナイザーの保存
config = AutoConfig.from_pretrained('再事前学習したモデルのpath')
tokenizer.save_pretrained('model_tokenizer')
config.save_pretrained('model_tokenizer')
