### Pretrainedモデルの再事前学習のTips from [Kaggle Discussion of AI4Code](https://www.kaggle.com/competitions/AI4Code/discussion/335294)

##### 　・Pre-train for as long as you can
##### 　・More data is better than more epochs
##### 　・Masking 15% of the tokens seems to be the sweet spot
##### 　・Use an MLM probability schedule (e.g. start with 0.15 and decrease to 0.05 over the course of training)
##### 　・Don't forget to use a warmup scheduler (linear warmup is fine)
##### 　・Always use an optimizer with weight decay (e.g. AdamW)
##### 　・If you are doing a question-answering task, you may want to use a different learning rate for the final linear layer (e.g. 0.01)
##### 　・When fine-tuning, always use a small learning rate (e.g. 1e-5)
##### 　・Don't forget to set the number of training epochs
##### 　・If you are using a TPU, use gradient accumulation (e.g. accumulate_grad_batches=8)
##### 　・If you are using a TPU, you may want to set a different batch size for each step (e.g. batch_size_per_step=batch_size_per_device *　gradient_accumulation_steps)
##### 　・For text classification, use DataCollatorForClassification
##### 　・For question answering, use DataCollatorForQuestionAnswering
##### 　・For sequence tagging, use DataCollatorForPermutationLanguageModeling
##### 　・For next sentence prediction, use DataCollatorForNextSentencePrediction
##### 　・For language modeling, use DataCollatorForLanguageModeling
##### 　・For permutation language modeling use DataCollatorForPermutationLanguageModeling
