from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorForLanguageModeling

dataset = Dataset.from_pandas(train_df)
model = AutoModel.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
      output_dir="./output-mlm",
      evaluation_strategy="epoch",
      learning_rate=args.lr,
      weight_decay=0.01,
      save_steps=10_000,
      save_strategy='no',
      per_device_train_batch_size= 64,
      num_train_epochs= 1,
      # report_to="wandb",
      run_name=f'output-mlm-{args.exp_num}',
      # logging_dir='./logs',
      lr_scheduler_type='linear',
      warmup_ratio=0.2,
      logging_steps=500,
      gradient_accumulation_steps= 4,
      overwrite_output_dir=True,
)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets['valid'],
        data_collator=data_collator,
        # optimizers=(optimizer, scheduler)
    )

    trainer.train()
