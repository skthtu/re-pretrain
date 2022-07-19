model = AutoModelWithLMHead.from_pretrained('prajjwal1/bert-small')


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

dataset= LineByLineTextDataset(
    tokenizer = tokenizer,
    file_path = './text.txt',
    block_size = 128  # maximum sequence length
)

print('No. of lines: ', len(dataset)) # No of lines in your datset

training_args = TrainingArguments(
    output_dir='./',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    save_steps=10000,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model('./')
