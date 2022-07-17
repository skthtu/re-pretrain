from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorForLanguageModeling

dataset = Dataset.from_pandas(train_df)
model = AutoModel.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        save_steps=10_000,
        save_total_limit=2,
    ),
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
